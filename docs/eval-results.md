# RegRAG Evaluation Results

Latest run: **2026-05-08** against the 28-question seed set. System under test: voyage-3.5-lite (512-dim) embeddings + claude-haiku-4-5 (classifier + inline citation judge) + claude-sonnet-4-6 (decomposer + synthesizer + offline judge).

Raw reports: [eval-20260508-184200.json](../packages/eval/results/eval-20260508-184200.json) (current) · [eval-20260507-044907.json](../packages/eval/results/eval-20260507-044907.json) (baseline, before inline judge).

---

## Headline metrics

| Metric | Score | Δ vs. baseline | Methodology |
|---|---|---|---|
| **Retrieval recall** | **98.3%** | — | Macro-averaged over 20 answer-expected questions: fraction of the question's `expected_passages_keywords` that appear (case-insensitive substring) in any retrieved chunk |
| **Refusal accuracy** | **89.3%** | — | 25 of 28 questions correctly refused-when-OOS or answered-when-in-scope. Two improvements + two regressions vs. baseline (see "Refusal shifts" below) |
| **Citation faithfulness** | **91.2%** | **+20.9 pp** | LLM-as-judge (Sonnet, offline) scores each `[[chunk_id]]` citation against the chunk it's attributed to; macro-averaged over all cited claims |

The citation-faithfulness gain is the result of a runtime intervention: a second-step judge runs inline as part of `verify` (Haiku, ~$0.005 per chat call, ~3s added latency) and strips claims whose citations don't substantively support them. See "Architecture change driving the CF gain" below.

## Per-persona breakdown

| Persona | n | Refusal | Recall | Citation faithfulness | CF Δ |
|---|---|---|---|---|---|
| compliance_analyst | 7 | **100%** | **100%** | 83.3% | +11.4 pp |
| counsel | 7 | 85.7% | 93.3% | 91.2% | +9.8 pp |
| federal_staff | 7 | 85.7% | **100%** | 93.8% | **+28.4 pp** |
| researcher | 7 | 85.7% | **100%** | **95.3%** | **+30.5 pp** |

The personas with the lowest baseline CF (federal_staff and researcher) saw the biggest gains. Their questions had the most claims that paraphrased commenter views or jurisdictional discussion as Commission rulings — exactly the failure mode the inline judge targets.

## Architecture change driving the CF gain

The baseline verifier did one check: do the cited `chunk_id`s exist in the retrieved set? That catches hallucinated citations but doesn't catch *misattributed* citations — claims that point to a real chunk that just doesn't say what the model claims it says. The eval surfaced this as the dominant failure pattern (~30% of cited claims).

The new verifier adds a second step. After the chunk-id presence check passes, a small Haiku call scores each (sentence, cited_chunk) pair 0/1 for substantive support. The same methodology the offline Sonnet judge uses — just runtime, cheaper, and with action: drop unsupported sentences from the final answer; drop individual citations from sentences with a mix; if the strip rate exceeds 50%, regenerate (bounded by the existing 2-attempt cap).

Code: [`apps/api/src/regrag_api/verification/substantive.py`](../apps/api/src/regrag_api/verification/substantive.py). Wired into the verify node in [`nodes/verify.py`](../apps/api/src/regrag_api/orchestration/nodes/verify.py).

## Refusal shifts vs. baseline

Same 89.3% in aggregate, but four questions changed behavior:

| Question | Expected | Baseline | Now | Direction |
|---|---|---|---|---|
| counsel-003 | answer | wrong (refused) | **correct (answered)** | improvement |
| researcher-oos-002 | refuse | wrong (answered) | **correct (refused)** | improvement |
| counsel-005 | answer | correct (answered) | wrong (refused) | regression |
| fedstaff-001 | answer | correct (answered) | wrong (refused) | regression |

The two regressions are the cost of the more aggressive verifier: on questions where the chunks are weakly aligned with the claim, the substantive judge strips most sentences, the system regenerates twice with no improvement, and what's left ends up empty enough to look like a refusal. Tuning levers if these become a pattern: raise the strip-rate-→-regen threshold from 50% to 70%, improve the synthesis prompt to pick chunks more deliberately, or accept partial answers rather than refusing on heavily-stripped drafts.

## Where the system fails

Three patterns surfaced repeatedly. None invalidate the architecture; all are eval-driven targets for the next iteration.

1. **Citation drift toward topical chunks** (drives the 30% citation-faithfulness gap). The model often cites chunks that are *about* the right topic but don't contain the *specific assertion* the answer makes. A common failure: citing a chunk that discusses commenter views ("Some commenters argue X") when the claim is about the Commission's ruling ("FERC requires X"). The chunk is on-topic; the chunk is not the substantive support.

2. **Borderline OOS gets qualified answers instead of refusal** (3/8 OOS misjudged). "How does FERC compare to state PUCs?" pulled jurisdiction-discussion chunks and constructed a hedged comparison instead of refusing. The pre-generation cosine threshold (0.35) caught nothing here; the LLM's `refused: true` flag is the only mechanism, and it was lenient. Tightening the threshold would also block real identifier-heavy queries — better levers are a stricter system prompt or a separate "is the question answerable from these chunks" classifier.

3. **Multi-doc latency at p90 ~33s, max ~42s.** Synthesis dominates (12–15s of that, sometimes more). At the edge of acceptable chat UX. Mitigations: streaming Sonnet tokens (currently the synthesis stage is "complete" only at the end), or routing simpler multi-doc questions to a smaller model.

## Latency distribution

| Stat | Value |
|---|---|
| min | 4.5s |
| median | 13.9s |
| p90 | 33.5s |
| max | 41.6s |
| single-doc avg | 10.1s |
| multi-doc avg | 22.4s |

## Classifier behavior

- 13/28 (46%) routed `single_doc`, 15/28 (54%) routed `multi_doc`
- Accuracy on the labeled subset: **17/20 = 85%**
- Misclassifications all routed *toward* the agentic path (false-multi-doc), which costs latency + a Sonnet call but doesn't degrade answer quality

## What the eval does not measure

- **Factual accuracy of the answer text itself.** The judge scores whether each cited chunk supports the claim it's attached to; it does not separately assess whether the claim is correct. Claims with wrong attribution can still be factually right; claims with right attribution can still be wrong.
- **Adversarial robustness.** All 28 questions are good-faith. We have not tested prompt-injection, jailbreak attempts, or queries designed to confuse the classifier.
- **Long-tail performance.** 28 questions is small enough to author by hand; large enough to detect regressions on the seed personas; too small to characterize the long tail of real-user questions a deployed system would see. The set should grow as the system is used.

## How to reproduce

```bash
cd ~/regrag
packages/eval/.venv/bin/regrag-eval run
```

Cost: roughly $2–4 in Anthropic spend for a full run (~28 chat invocations + ~20 judge invocations on Sonnet). Wall-clock ~12 minutes. Results land in `packages/eval/results/eval-{timestamp}.json` with full per-question breakdown including the judge's per-claim rationale.

To run a smaller slice:

```bash
regrag-eval run --filter "compliance-001,counsel-001,researcher-oos-001"
regrag-eval run --persona compliance_analyst
regrag-eval run --no-judge   # skip LLM-as-judge, faster + cheaper
```
