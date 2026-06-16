# Keystone Experiment — Pre-Registration

**Branch:** session1/honest-wiring-and-energy-gating
**Location:** experiments/keystone/ (clean room — must NOT import zero_defect_shell, ruliad_engine, mfu_v2, stage9)
**Status:** registered BEFORE results are seen. Do not edit thresholds after running.

---

## The one question

Can the trained energy head, reading the hidden-state representation of a **fully
decoded** answer, rank correct answers above incorrect ones on a benchmark it did
not help design?

This is **Version A (reranking)** — the prerequisite for the roadmap's
"score-before-decode" efficiency claim (Version B). A does not save compute. A is
the gate: if A fails, B cannot work, and the latent-scoring core of the
architecture is unproven.

## Substrate (fixed)

- **Base model:** `_MODEL_NAME` from config.py  (≈1.5B open-weights)
- **Benchmark:** GSM8K, `test` split (grade-school math, exact-match — answer is objectively checkable)
- **N questions:** 300  (pre-committed; do not cherry-pick)
- **K samples/question:** 8   (sample 0 = greedy; 1–7 = temperature 0.8)
- **Prompt:** fixed 4-shot chain-of-thought, identical across all conditions
- **Correctness:** exact match of the final integer (GSM8K `####` gold)
- **Seed:** fixed and recorded

## Conditions (all evaluated on the SAME fixed K candidates per question)

| name | selection rule |
|---|---|
| avg single-sample | expected accuracy of one random draw (= random pick) |
| majority vote | most common parsed answer over the K samples (self-consistency) |
| **energy-best** | the candidate the energy head scores best (lowest energy) |
| oracle pass@K | correct if ANY of the K candidates is correct (ceiling) |

Plus **energy AUC**: probability that a randomly chosen correct candidate gets a
better energy score than a randomly chosen incorrect one. 0.5 = no signal.

## Decision rule (FIXED — checked by analyze.py)

Let CIs be 95% bootstrap intervals over questions.

- **STRONG / build Version B:** energy-best ≥ majority vote (point estimate) AND
  energy AUC ≥ 0.60 with CI excluding 0.5. → The foundation holds. Proceed to the
  score-before-decode test.
- **USEFUL / keep going:** energy-best beats avg single-sample with non-overlapping
  CIs, even if it does not beat majority vote. → Real signal; iterate the head.
- **NO SIGNAL:** energy AUC CI includes 0.5 AND energy-best CI overlaps avg
  single-sample. → The head has no real discriminative power on external data.
  Two honest retries allowed: (1) retrain the head on GSM8K-derived correct/incorrect
  pairs; (2) change the representation it reads. Then decide continue vs pivot.
- **ANTI-CORRELATED:** energy AUC < 0.45. → Sign/convention bug or inverted signal.
  Debug before concluding anything.

## Out of scope for this experiment (do NOT claim from it)

- Any compute / energy / token *savings*. This test decodes all K candidates; it
  measures selection quality only. Savings belong to Version B and require a
  separate test that counts the wrapper's OWN compute (branching + energy + routing),
  not just decoded tokens.
- Anything about "beating frontier models." Irrelevant here. The comparison is
  base-vs-base+head on one benchmark.

## What gets written up either way

All four condition accuracies + AUC + CIs, including disappointing ones. A null
result is a result and gets recorded, not buried.
