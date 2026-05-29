# RegRAG Evaluation Results

Latest production run: **2026-05-29** (v6) against the **40-question** eval set, now over a **17-document** corpus. System under test: voyage-3.5-lite (512-dim) embeddings + claude-haiku-4-5 (classifier + inline citation judge) + claude-sonnet-4-6 (decomposer + synthesizer + offline judge). v6 adds the citation-attribution hardening from the post-review pass: document-anchored retrieval (per-accession quota + per-accession RRF), the per-sentence accession-scope verifier, and two new Order 841 sources (the rehearing Order 841-A and the Federal Register publication carrying the literal effective date).

Raw reports:
- [eval-20260529-144731.json](../packages/eval/results/eval-20260529-144731.json) — **v6** (40 questions, 17-doc corpus, doc-anchored retrieval + scope verifier)
- [eval-20260509-041345.json](../packages/eval/results/eval-20260509-041345.json) — **v5** (40 questions, post corpus expansion to 15 docs)
- [eval-20260508-184200.json](../packages/eval/results/eval-20260508-184200.json) — **v2/v3** (28 questions, 8-doc corpus, inline judge)
- [eval-20260508-203704.json](../packages/eval/results/eval-20260508-203704.json) — **v4** (structured-output synthesis + quote verification; reverted, see "What didn't work" below)
- [eval-20260508-221716.json](../packages/eval/results/eval-20260508-221716.json) — **thin baseline** (vanilla RAG, minimal prompt)
- [eval-20260508-230246.json](../packages/eval/results/eval-20260508-230246.json) — **matched baseline** (RegRAG prompt, no agentic, no verification)
- [eval-20260507-044907.json](../packages/eval/results/eval-20260507-044907.json) — **v1 baseline** (chunk-id verifier only)

---

## Headline metrics (v6, 40 questions, 17-doc corpus)

| Metric | Score | v5 → v6 | Methodology |
|---|---|---|---|
| **Retrieval recall** | **95.8%** | 96.9% → 95.8% (−1.1pp) | Macro-averaged over answer-expected questions: fraction of the question's `expected_passages_keywords` that appear (case-insensitive substring) in any retrieved chunk |
| **Refusal accuracy** | **92.5%** | 90.0% → 92.5% (+2.5pp) | 37 of 40 questions correctly refused-when-OOS or answered-when-in-scope |
| **Refusal precision** | **77.8%** | 70.0% → 77.8% (+7.8pp) | Of 9 refusals, 7 were correct (2 false positives on should-have-answered questions) |
| **Refusal recall** | **87.5%** | 87.5% → 87.5% (flat) | Of 8 should-refuse questions, 7 were refused (1 false negative — see below) |
| **Citation faithfulness** | **94.8%** | 95.4% → 94.8% (−0.6pp) | LLM-as-judge (Sonnet, offline) scores each `[[chunk_id]]` citation against the chunk it's attributed to; macro-averaged over all cited claims |

**Honest read of the v5 → v6 delta.** The aggregate moved within noise on the citation/recall axes and improved on refusal. The CF dip (−0.6pp) is inside the run-to-run variance of a non-deterministic Sonnet judge — at the question level CF rose on five questions and fell on five; nothing systematic. Refusal accuracy and precision both improved. The recall dip (−1.1pp) is **one question** — `counsel-005` — and is a metric artifact, not a behavior change (detail below). The headline functional win isn't in this table: the canonical citation-misattribution case ("effective date of Order 841") now answers with the literal date **June 4, 2018** cited to Order 841's own Federal Register text, verified in production — see "Citation-attribution hardening (v6)" below.

The precision/recall split is more honest than the headline accuracy. The two baselines below both score ~71% refusal accuracy not because they correctly refuse OOS questions — they don't refuse anything ever — but because they correctly answer the should-answer questions, which makes the binary accuracy look respectable. Splitting precision/recall exposes that.

For reference, v3 on the original 28-question set hit 89.3% refusal / 98.3% recall / 91.2% CF. **These are not directly comparable to v6** — different eval sets, different corpora — but the order-of-magnitude consistency suggests the architecture generalizes rather than overfits to the original test bank.

**Important methodological caveat for the headline numbers:** all 40 questions were authored by the same person who designed the system, so the eval exercises queries the system is biased toward handling well. The numbers should be read as "this is what the system does on author-curated questions"; real-user query distributions will look different. See "What the eval does not measure" below for the full list of confounds.

## Per-persona breakdown

| Persona | n | Refusal | Recall | Citation faithfulness |
|---|---|---|---|---|
| **compliance_analyst** | 10 | **100%** | **100%** | **98.2%** |
| counsel | 10 | 90.0% | 87.5% | 94.4% |
| federal_staff | 10 | 90.0% | 95.8% | 94.0% |
| researcher | 10 | 90.0% | **100%** | 92.4% |

compliance_analyst is now the strongest persona (100% refusal and recall, 98.2% CF) — its single-document obligation-lookup questions are exactly what doc-anchored retrieval helps most. counsel is the weakest on recall (87.5%), pulled down entirely by `counsel-005` (the D.C. Circuit question — see below). The researcher persona dropped from v5's 100% refusal to 90% because of `researcher-oos-002` flipping to a (soft) answer; its recall stayed at 100%.

## Citation-attribution hardening (v6)

v6 targets one specific, demonstrated failure: when a query names an order, the retriever could surface chunks that *reference* that order from a different document and the synthesizer would cite them as if they were *from* the named order. The canonical case — "What was the effective date of Order 841?" — cited Order 845-A (a different rulemaking that mentions 841's deadline in passing) and never stated an actual date.

Three coordinated changes:

1. **Document-anchored retrieval** — when `extract_identifiers` finds a named order, a separate vector search restricted to that order's accession(s) joins the RRF pool, with a **per-accession quota** (each in-scope accession gets its own top-K slots) and **per-accession RRF ranking** (each accession's #1 contributes equal weight, so a third in-scope source isn't buried at global rank 13). [`retrieval/hybrid.py`](../apps/api/src/regrag_api/retrieval/hybrid.py)
2. **Per-sentence accession-scope verifier** — post-synthesis, for each sentence whose subject is the named order, the *first* citation must be from an in-scope accession; out-of-scope first-citations are stripped (a later in-scope citation promotes) or the sentence is dropped. [`verification/scope.py`](../apps/api/src/regrag_api/verification/scope.py)
3. **Two new Order 841 sources** — the rehearing **Order 841-A** (`20190516-3057`) and the **Federal Register** publication (`fr-2018-03-06-2018-03708`), the latter parsed with column-aware extraction so its "effective June 4, 2018" paragraph survives as clean text. (Closes the deferred "corpus missing 841-A" review finding.)

**Result, verified in production:** the canonical query now answers *"Order 841 was issued on February 15, 2018 `[[20180228-3066:c0000]]`. According to the Federal Register publication of the Final Rule, the effective date was June 4, 2018 `[[fr-2018-03-06-2018-03708:c0274]]`."* — both citations in the Order 841 family, the literal correct date, no 845-A misattribution.

### The two question-level wrinkles (honest accounting)

The aggregate hides two question-level changes; both are minor and neither is a behavior regression caused by the verifier:

- **`counsel-005` ("Did the D.C. Circuit uphold Order 841, and on what grounds?"): recall 0.667 → 0.0.** This is the entire −1.1pp aggregate recall drop. It is a **retrieval-metric artifact, not a behavior change** — the question refuses in *both* v5 and v6 (it refused in the baseline too). What changed: anchoring on "Order 841" now floods the top-10 with the three 841-family documents, displacing the Order 2222 chunk that previously contributed the keyword hits the recall metric counts. Compounding it, the question is partly unanswerable from the corpus: the D.C. Circuit opinion (*NARUC v. FERC*, July 2020) **isn't in the corpus**, so "on what grounds" was never fully groundable. Refusing it is arguably the *correct* regulated-domain behavior. Two honest fixes exist (lighten anchoring on multi-doc/cross-order queries, or add the court opinion to the corpus); neither is blocking.
- **`researcher-oos-002` ("impact of FERC orders on retail consumer electricity prices?"): refused → answered.** A genuine but soft over-answer regression. The corpus addresses *wholesale* markets, not *retail* consumer prices; the right behavior is to decline. v6 instead pivots to grounded wholesale/Order 745 LMP content rather than refusing the retail question — the answer is faithfully cited, just off-target. Likely a side-effect of the K 8→10 retrieval bump giving the synthesizer enough material to attempt an answer. This is the precision/recall tension the doc-anchored change surfaces: more aggressive retrieval lowers refusal precision on borderline-OOS questions.

The net is favorable — refusal accuracy +2.5pp, refusal precision +7.8pp, the canonical misattribution fixed — but the wrinkles are real and named here rather than smoothed over.

## Answerability gate experiment (issue #3, 2026-05-29)

**Motivation.** v6 surfaced `researcher-oos-002` (a retail-price jurisdiction question) flipping from refuse to a soft over-answer, and refusal calibration was the #1 named gap. Hypothesis: a dedicated pre-synthesis *answerability* check — a Haiku call judging whether the retrieved chunks actually support the question as asked — would catch out-of-scope questions the synthesizer's own `refused: true` self-flag misses, especially the hard ones that name an in-corpus order (so retrieval returns high-cosine chunks and naive relevance can't tell they're unanswerable).

**Expanded test set.** The OOS set grew from 8 to 23 questions, tagged by *kind* of unanswerability so recall can be broken down by category (total eval set 40 → 55):

| category | n | what it tests |
|---|---|---|
| `topic_absent` | 8 | topic simply not in corpus (rooftop solar, pipelines, cybersecurity) |
| `false_premise` | 6 | names an in-corpus order but asks for a fact it doesn't establish ("how much must distribution utilities pay aggregators under Order 2222?") |
| `order_conflation` | 5 | attributes one order's subject to another ("Order 841's interconnection queue reforms" — 841 is storage) |
| `jurisdiction_boundary` | 4 | asks about something outside federal-wholesale scope (retail rates, net metering, GHG reporting) |

**A/B result** (55 questions, `--no-judge`; the gate only adds refusals, so refusal metrics are what move):

| Policy | Refusal precision | Refusal recall | False-refused in-scope | Refusal accuracy |
|---|---|---|---|---|
| Self-flag (gate off) | **90.9%** | 87.0% (20/23) | 2 | 90.9% |
| Gate, **all** queries | 63.9% | **100%** (23/23) | 13 | 76.4% |
| Gate, **single_doc** only | 84.6% | 95.7% (22/23) | 4 | 90.9% |

Gating *all* queries reaches 100% recall but collapses precision — it false-refuses 9 multi-doc synthesis questions, because on a cross-document question each retrieved chunk only partially covers the ask and the gate misreads distributed-but-present support as "unanswerable." Restricting the gate to `single_doc`-classified queries fixes that: it lands at **identical accuracy** to the self-flag baseline (90.9%, 50/55 both) — a pure rebalance, trading **−6.3pp precision for +8.7pp recall**. It converts 2 missed OOS questions into catches and creates 2 new false-refusals (`counsel-003`, `compliance-004`), a literal wash on count. (It's also sensitive to classifier error: a question mislabeled `single_doc` gets gated and can be over-refused.)

**The decisive per-category finding:**

The decisive per-category finding:

| category | self-flag recall | gate (single_doc) recall |
|---|---|---|
| topic_absent | 7/8 | 7/8 |
| false_premise | **6/6** | 6/6 |
| order_conflation | **5/5** | 5/5 |
| jurisdiction_boundary | 2/4 | **4/4** |

The synthesizer's existing self-flag **already catches every false_premise and order_conflation question** — the exact categories the gate was built to catch. Handed chunks that don't support a false-premise or conflated question, the synthesizer already declines. The gate's contribution is concentrated entirely in `jurisdiction_boundary`, where it closes the gap fully (2/4 → 4/4) — including the originally-motivating `researcher-oos-002` retail-price question. `topic_absent` stays 7/8 (the one miss is a multi-doc OOS the gate doesn't run on).

**Conclusion — an accuracy-neutral rebalance, kept off by default.** Restricted to `single_doc` queries, the gate is a *wash on accuracy* (90.9% = 90.9%): it trades 6.3pp of refusal precision for 8.7pp of recall, fully closing the jurisdiction-boundary gap but adding two false-refusals of answerable questions. Whether that trade is worth it is domain-dependent — in a regulated setting that prizes "I don't know" over a confident wrong answer, recall-favoring is defensible; on raw accuracy it's neutral. So the gate ships behind `REGRAG_ANSWERABILITY_GATE` (default **off**) and is preserved as a documented, toggle-able experiment (like the v4 structured-output verification): **enable it if a deployment's query mix is jurisdiction-heavy and recall-favoring; leave it off otherwise.** The two findings that matter most regardless: (1) the existing self-flag is already well-calibrated for the dangerous false-premise/conflation categories — no gate needed there; (2) the durable wins are the **expanded 23-question OOS set** and **per-category recall tagging**, which stay in the harness and make future refusal work measurable.

## Architecture change driving the CF gain

The baseline verifier did one check: do the cited `chunk_id`s exist in the retrieved set? That catches hallucinated citations but doesn't catch *misattributed* citations — claims that point to a real chunk that just doesn't say what the model claims it says. The eval surfaced this as the dominant failure pattern (~30% of cited claims).

The new verifier adds a second step. After the chunk-id presence check passes, a small Haiku call scores each (sentence, cited_chunk) pair 0/1 for substantive support. The same methodology the offline Sonnet judge uses — just runtime, cheaper, and with action: drop unsupported sentences from the final answer; drop individual citations from sentences with a mix; if the strip rate exceeds 50%, regenerate (bounded by the existing 2-attempt cap).

Code: [`apps/api/src/regrag_api/verification/substantive.py`](../apps/api/src/regrag_api/verification/substantive.py). Wired into the verify node in [`nodes/verify.py`](../apps/api/src/regrag_api/orchestration/nodes/verify.py).

## Refusal shifts (historical: v5 vs. v1 baseline)

*This section documents the v1 → v5 verifier transition and is kept for history; the v5 → v6 shifts are covered under "Citation-attribution hardening (v6)" above.*

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

| Setup | Refusal acc | Refusal P / R | Retrieval recall | Citation faithfulness |
|---|---|---|---|---|
| **Thin baseline** — pure vector top-10 + plain Sonnet, minimal prompt | 71.4% | — / **0%** | 98.3% | **5.0%** |
| **Matched baseline** — same retrieval + Sonnet, but with RegRAG's prompt discipline (citation format + "find the supporting phrase or drop the claim") | 71.4% | — / **0%** | 98.3% | **81.1%** |
| **RegRAG (full)** — agentic decomposition + hybrid retrieval + chunk-id verifier + Haiku substantive judge | **89.3%** | **77.8% / 87.5%** | 98.3% | **91.2%** |

The "—" in the Refusal P/R column for the baselines means precision is undefined — they emitted zero refusals across all 28 questions, so there's nothing to compute precision against. Both baselines correctly answered the 20 should-answer questions (giving them ~71% accuracy on the binary refusal metric) but never declined a single should-refuse question (0% recall). The RegRAG architecture is what gives the system a working refusal mechanism at all.

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
