"""regrag-eval CLI."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

from .runner import load_eval_set, report_to_dict, run_all


def _repo_root() -> Path | None:
    # parents: [0]=regrag_eval [1]=src [2]=eval [3]=packages [4]=regrag (repo root).
    # Returns None inside containers where the path is shorter (the eval CLI is
    # local-only — eval invokes the production graph via the regrag-api package,
    # but the CLI itself runs from a dev shell).
    try:
        return Path(__file__).resolve().parents[4]
    except (IndexError, OSError):
        return None


def _load_env():
    root = _repo_root()
    if root is None:
        return
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


def _default_eval_set_path() -> Path:
    return Path(__file__).parent / "eval_set.yaml"


def _default_output_dir() -> Path:
    root = _repo_root()
    base = root if root is not None else Path.cwd()
    return base / "packages" / "eval" / "results"


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase log verbosity")
def cli(verbose: int):
    level = logging.WARNING - (10 * verbose)
    logging.basicConfig(level=max(logging.DEBUG, level), format="%(levelname)s %(name)s: %(message)s")
    _load_env()


@cli.command()
@click.option(
    "--eval-set", "eval_set_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to eval_set.yaml (default: bundled)",
)
@click.option(
    "--filter", "id_filter", default=None,
    help="Run only the question with this id, or comma-separated ids",
)
@click.option(
    "--persona", default=None,
    help="Run only questions for this persona",
)
@click.option("--no-judge", is_flag=True, help="Skip the LLM-as-judge citation faithfulness step (faster, cheaper)")
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the JSON report (default: packages/eval/results/)",
)
def run(eval_set_path, id_filter, persona, no_judge, out_dir):
    """Run the eval and produce a report."""
    if eval_set_path is None:
        eval_set_path = _default_eval_set_path()
    questions = load_eval_set(eval_set_path)

    if id_filter:
        wanted = {s.strip() for s in id_filter.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if persona:
        questions = [q for q in questions if q["persona"] == persona]

    if not questions:
        click.echo("No questions match the filter.", err=True)
        sys.exit(1)

    click.echo(f"Running {len(questions)} questions (judge={'off' if no_judge else 'on'})...\n")

    def progress(i, n, qid):
        click.echo(f"  [{i}/{n}] {qid}", err=True)

    report = run_all(questions, run_judge=not no_judge, progress_callback=progress)

    # Write JSON report
    if out_dir is None:
        out_dir = _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"eval-{timestamp}.json"
    out_path.write_text(json.dumps(report_to_dict(report), indent=2, default=str))

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("EVAL REPORT")
    click.echo("=" * 60)
    s = report
    click.echo(f"Questions: {s.n_questions} ({s.n_errors} errored)")
    click.echo(f"Refusal accuracy:        {_pct(s.refusal_accuracy)}")
    click.echo(f"Retrieval recall:        {_pct(s.retrieval_recall_macro)}  (answer-expected only)")
    click.echo(f"Citation faithfulness:   {_pct(s.citation_faithfulness_macro)}  (LLM-as-judge)")
    click.echo("\nPer persona:")
    for p, m in sorted(s.by_persona.items()):
        click.echo(
            f"  {p:>20} (n={m['n']}): "
            f"refusal={_pct(m['refusal_accuracy'])}  "
            f"recall={_pct(m['retrieval_recall'])}  "
            f"cf={_pct(m['citation_faithfulness'])}"
        )
    click.echo(f"\nFull report: {out_path}")

    # Return non-zero if any question errored — useful for CI
    if s.n_errors:
        sys.exit(1)


def _pct(v: float | None) -> str:
    if v is None:
        return "  n/a "
    return f"{v * 100:5.1f}%"


if __name__ == "__main__":
    cli()
