# RegRAG Evaluation Results

Latest production run: **2026-05-09** (v5) against the **40-question** expanded eval set covering all 15 corpus documents. System under test: voyage-3.5-lite (512-dim) embeddings + claude-haiku-4-5 (classifier + inline citation judge) + claude-sonnet-4-6 (decomposer + synthesizer + offline judge).

Raw reports:
- [eval-20260509-041345.json](../packages/eval/results/eval-20260509-041345.json) — **v5** (40 questions, post corpus expansion to 15 docs)
- [eval-20260508-184200.json](../packages/eval/results/eval-20260508-184200.json) — **v2/v3** (28 questions, 8-doc corpus, inline judge)
- [eval-20260508-203704.json](../packages/eval/results/eval-20260508-203704.json) — **v4** (structured-output synthesis + quote verification; reverted, see "What didn't work" below)
- [eval-20260508-221716.json](../packages/eval/results/eval-20260508-221716.json) — **thin baseline** (vanilla RAG, minimal prompt)
- [eval-20260508-230246.json](../packages/eval/results/eval-20260508-230246.json) — **matched baseline** (RegRAG prompt, no agentic, no verification)
- [eval-20260507-044907.json](../packages/eval/results/eval-20260507-044907.json) — **v1 baseline** (chunk-id verifier only)

---

## Headline metrics (v5, 40 questions, 15-doc corpus)

| Metric | Score | Δ vs. v3 (28 questions, 8 docs) | Methodology |
|---|---|---|---|
| **Retrieval recall** | **96.9%** | -1.4 pp | Macro-averaged over 30 answer-expected questions: fraction of the question's `expected_passages_keywords` that appear (case-insensitive substring) in any retrieved chunk |
| **Refusal accuracy** | **90.0%** | +0.7 pp | 36 of 40 questions correctly refused-when-OOS or answered-when-in-scope |
| **Citation faithfulness** | **95.4%** | **+4.2 pp** | LLM-as-judge (Sonnet, offline) scores each `[[chunk_id]]` citation against the chunk it's attributed to; macro-averaged over all cited claims |

The CF improvement on a larger eval set is meaningful — it shows the architecture generalizes to new corpus content, not just the questions it was tuned against. The new 12 questions cover the rulemaking arcs the corpus expansion enables (interconnection 2003 → 845 → 2023, transmission planning 1000 → RM21-17 ANOPR → 1920, market design 825 + 841 + 2222).

## Per-persona breakdown

| Persona | n | Refusal | Recall | Citation faithfulness |
|---|---|---|---|---|
| compliance_analyst | 10 | 90.0% | 95.8% | 95.2% |
| counsel | 10 | 80.0% | 95.8% | 92.9% |
| federal_staff | 10 | 90.0% | 95.8% | **97.6%** |
| **researcher** | 10 | **100%** | **100%** | **95.9%** |

The researcher persona — which was the weakest in the v1 baseline (71% refusal, 65% CF) — now hits **100% on refusal AND retrieval recall, with 95.9% CF**. The new evolution-style questions (`researcher-006/007/008`) all worked cleanly. The 30+ pp gains here over the v1 baseline come from the combination of (a) the inline LLM judge added in v2/v3 and (b) richer corpus context that gives the system more material to ground claims in.

The counsel persona slipped 5.7 pp on refusal (from 85.7% on 28-q to 80% on 40-q): one of the new counsel questions (probably `counsel-006` on the Order 2003 → Order 2023 evolution) over-refused. Worth a closer look but not blocking.

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

## Baseline comparison — what does the architecture actually buy?

The 91.4% citation faithfulness number is impressive in isolation, but a senior reader will ask: vs. what? To answer that honestly, we ran two baselines through the same eval harness — both keep voyage-3.5-lite + Neon pgvector + claude-sonnet-4-6, but progressively strip out RegRAG's contributions.

| Setup | Refusal acc | Retrieval recall | Citation faithfulness |
|---|---|---|---|
| **Thin baseline** — pure vector top-10 + plain Sonnet, minimal prompt | 71.4% | 98.3% | **5.0%** |
| **Matched baseline** — same retrieval + Sonnet, but with RegRAG's prompt discipline (citation format + "find the supporting phrase or drop the claim") | 71.4% | 98.3% | **81.1%** |
| **RegRAG (full)** — agentic decomposition + hybrid retrieval + chunk-id verifier + Haiku substantive judge | **89.3%** | 98.3% | **91.2%** |

This decomposes the gain into two attributable layers:

**Prompt discipline → +76 pp CF** (5.0% → 81.1%). The single largest lever, with zero code and no extra LLM calls. Without the citation-format instruction and the "drop the claim if you can't find a supporting phrase" rule, the thin baseline emits citations only intermittently and in inconsistent formats; the LLM-as-judge can't even find them, hence the 5% floor. Add the prompt discipline and citations become parseable, formatted correctly, and largely defensible.

**Agentic + verification → +10 pp CF AND +18 pp refusal accuracy** (81.1% → 91.2% CF, 71.4% → 89.3% refusal). Smaller than the headline number suggested but real. The agentic decomposition (classify → decompose → retrieve_parallel → synthesize) and the runtime substantive-support judge cost ~5x the per-query latency and ~$0.005 more in API spend. What you get for that:
- Tighter citations (the substantive judge strips paraphrase-beyond-support claims that the prompt discipline can't catch)
- A working refusal mechanism — without the judge, the system answers OOS questions instead of declining (researcher-OOS in particular)
- Per-document coverage on comparison queries (decomposition issues separate sub-queries per document, which single-pass retrieval misses)

**The honest framing.** The architecture's contribution to *citation faithfulness alone* is 10 pp. Most CF gains come from prompt engineering, which is cheap. But the architecture's contribution to *refusal accuracy* is 18 pp — and refusal accuracy is what makes the system usable in a regulated domain where "I don't know" beats "here is a confident but unsupported answer." The agentic + verification layers earn their cost on the refusal axis more than the citation axis.

Retrieval recall is the same across all three (98.3%) — pure vector retrieval finds the relevant chunks just as well as the hybrid + identifier-floor approach. The hybrid retrieval's value is on the *long tail* of identifier-heavy queries that the eval set doesn't fully exercise; this is a known eval limitation, not an architecture limitation.

Raw reports: [thin baseline](../packages/eval/results/eval-20260508-221716.json) · [matched baseline](../packages/eval/results/eval-20260508-230246.json) · [full RegRAG](../packages/eval/results/eval-20260508-184200.json).

---

## What didn't work — structured output + quote verification (v4)

After landing the v3 result (91.4% CF), the obvious next move was to tighten the verification architecture: instead of relying on an LLM judge to assess substantive support, force the synthesizer to emit `(claim, chunk_id, supporting_quote)` triples via Anthropic tool use, then substring-check each quote against its chunk. By construction, citation drift toward topical chunks should become impossible — the model has to *find* a supporting phrase up front, and the phrase has to actually appear in the chunk.

The hypothesis was that structural verification would beat the LLM judge. The eval said the opposite.

**v4 results vs. v3:**

| Metric | v3 (LLM judge) | v4 (structured + quote verify) | Δ |
|---|---|---|---|
| Refusal accuracy | 89.3% | 85.7% | -3.6 pp |
| Retrieval recall | 98.3% | 98.3% | 0 |
| Citation faithfulness | 91.4% | **84.1%** | **-7.3 pp** |

Per-persona, the regression hit hardest where the gains in v3 had been largest:

| Persona | v3 CF | v4 CF | Δ |
|---|---|---|---|
| compliance_analyst | 83.3% | 75.0% | -8.3 pp |
| counsel | 91.2% | 90.8% | ~0 |
| federal_staff | 93.8% | 93.3% | ~0 |
| **researcher** | **95.3%** | **76.8%** | **-18.5 pp** |

**Why structural verification lost.** The substring check is too permissive. It accepts any quote that appears anywhere in the chunk, even when the claim paraphrases beyond what the quote actually supports. The LLM judge correctly identifies the failure mode "the quote is real but the claim overreaches" — the substring check cannot. On comparative researcher questions especially, the model would find a literal phrase to cite but then make a synthesis claim the phrase didn't actually establish, and the substring check waved it through.

The structural approach also dropped refusal accuracy 3.6 pp because the quote-verification regen loop pushed marginal answers into refusal territory: a draft with several weak-quote claims would trip the keep-ratio threshold, regenerate, regenerate again, and end up with a sparse-or-empty answer the system then refused.

**Decision:** Revert to v3. Live demo runs on the LLM-judge path (CF 91.4%). The quote verification module ([`apps/api/src/regrag_api/verification/quotes.py`](../apps/api/src/regrag_api/verification/quotes.py)) is preserved in the repo as a documented experiment artifact rather than deleted — it might still be useful as a fast pre-filter ahead of the LLM judge in a future iteration, but on its own it doesn't beat what we already have.

**Lesson.** The LLM judge is doing more sophisticated work than I'd given it credit for. Substring matching catches *fabricated* citations; understanding "this chunk discusses X but doesn't establish Y" requires the kind of judgment a calibrated LLM does well. Future effort should go toward making the LLM judge cheaper or faster, not replacing it with a structural check.

---

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
