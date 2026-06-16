# Keystone Results — Inference-Time Verification vs. Adaptive Compute on a 1.5B Model

**Model:** Qwen2.5-1.5B-Instruct · **Benchmark:** GSM8K (test) · **Hardware:** single T4 (Colab)
**Protocol:** 150 questions, K=6 samples/question (sample 0 greedy; 1–5 temperature 0.8, top-p 0.95),
max_new_tokens 400, seed 0 → 900 candidates. Pre-registered in `SPEC.md`; data and code committed.
**Reproducible commits:** candidates + scores `c82872d`, efficiency analysis `a84d7a3`.

Two questions were asked. The first has a negative answer; the second a positive one. Both
are reported in full, including the numbers that disappoint, because the negative result is
what makes the positive one credible.

---

## TL;DR

1. **Verifier selection does not beat self-consistency.** Across six inference-time verifiers
   (free and trained, scalar and hidden-state), none improved on plain majority vote for choosing
   the answer. The verifiers carry real but insufficient signal (AUC ≈ 0.71–0.74).
2. **Adaptive early-stopping recovers full accuracy at ~31% less compute.** Stopping sampling once
   the answers agree (margin ≥ 2) matches majority@6 accuracy using 4.12 samples instead of 6 —
   a statistically indistinguishable accuracy at roughly a third less compute, with no verifier and
   no training.

---

## Finding 1 — Verifier selection vs. self-consistency (NEGATIVE)

**Question:** can an inference-time verifier select a better answer from K samples than majority
vote does?

Baselines (95% bootstrap CI over questions):

| strategy | accuracy |
|---|---|
| avg single-sample (random pick) | 50.2% [44.3, 56.0] |
| **majority vote (self-consistency)** | **68.7% [60.7, 76.0]** ← bar |
| oracle pass@K (a correct answer exists in the 6) | 83.3% [77.3, 89.3] ← ceiling |

The entire space available to any selector is the gap between majority and oracle: **68.7 → 83.3,
about 14.6 points.** Six verifiers were tested. For each, the strongest selection strategy
(confidence-weighted self-consistency, CW-SC) was compared to majority vote via a paired bootstrap
of the per-question difference:

| verifier | AUC | CW-SC acc | CW-SC − majority |
|---|---|---|---|
| mean log-prob (free) | 0.708 [0.668, 0.745] | 64.0% | −4.7pp [−8.7, −0.7] |
| min log-prob (free) | 0.66 | 60.7% | −8.0pp [−12.7, −4.0] |
| bottom-2 log-prob (free) | 0.68 | 60.0% | −8.7pp [−13.3, −4.7] |
| last-span log-prob (free) | ~0.50 | 59.3% | −9.3pp [−14.7, −4.0] |
| trained, scalar features (CV) | 0.74 | 62.7% | −6.0pp [−10.7, −2.0] |
| trained, scalar + hidden (CV) | 0.73 | 66.7% | −2.0pp [−8.0, +3.3] |

Trained verifiers were cross-validated by question (GroupKFold), so these are leakage-free
out-of-fold estimates.

**Conclusion.** No verifier beats majority vote. Most are reliably behind (CIs entirely below 0);
the best (`both`) is a statistical tie that leans behind. The "sharper" free signals (min, bottom-2,
last-span) performed *worse*, falsifying the hypothesis that a single broken token reliably flags a
wrong answer. The convergence of six independent methods to the same narrow band below the baseline
is the signature of a structural ceiling, not an implementation gap: on GSM8K the correct answer,
when it exists in the sample set, is usually already the plurality — so the cases a verifier could
win are rare and are exactly the ones the model produced *least* confidently.

---

## Finding 2 — Adaptive early-stopping (POSITIVE)

**Question:** can we reach majority@K accuracy using *fewer* than K samples on average, by stopping
once the samples agree? No verifier — just counting agreement.

**Static saturation** (majority vote on the first *m* samples, averaged over 300 random orderings):

| samples used | accuracy |
|---|---|
| 1 | 50.3% [44.5, 56.1] |
| 2 | 50.3% [44.5, 56.1] |
| 3 | 56.4% [50.0, 62.8] |
| 4 | 60.4% [53.6, 67.0] |
| 5 | 63.6% [56.5, 70.3] |
| 6 | 66.8% [59.4, 73.9] ← target |

**Adaptive stopping** (draw one at a time; stop when the leading answer's margin over the runner-up
reaches L; cap at 6):

| policy | avg samples | accuracy | vs majority@6 |
|---|---|---|---|
| L=1 (stop at first lead) | 1.00 | 50.3% [44.5, 56.1] | −16.5pp [−20.7, −12.5] — below |
| **L=2** | **4.12** | **66.7% [59.3, 73.8]** | **−0.1pp [−0.3, +0.1] — matches** |
| L=3 | 5.00 | 66.8% [59.4, 73.9] | +0.0pp — matches |

**Conclusion.** Stopping at margin L=2 reaches **66.7%**, statistically indistinguishable from
majority@6's 66.8% (paired difference −0.1pp, CI [−0.3, +0.1]), at **4.12 average samples — a 31%
reduction in compute for the same accuracy.** The mechanism is simple and free: easy questions
settle in 2–3 samples and stop early; only hard questions spend the full budget. The too-aggressive
L=1 policy correctly collapses to single-sample accuracy, confirming the method also identifies what
does not work.

(Note: majority@6 reads 66.8% here vs. 68.7% in Finding 1. Same data, different estimator — Finding 1
takes the deterministic mode once per question; Finding 2 averages the mode over random orderings
with random tie-breaks. The ~2pp difference is tie-breaking variance and is within the CIs.)

---

## What this demonstrates

On a 1.5B model on GSM8K: inference-time verification does **not** improve *which* answer you select
over majority vote, but adaptive early-stopping recovers the **full** accuracy of majority voting at
roughly **a third less compute**. The useful lever for a small model is not a smarter judge of
answers — it is knowing when to stop asking.

## Scope and honest limits

- One model (1.5B), one benchmark (GSM8K), arithmetic reasoning only. Not claimed to generalize.
- K ≤ 6 throughout. This supports "can we stop *earlier* than 6"; it cannot address "would *more*
  than 6 help," which is a separate, more expensive question deliberately not asked.
- The verifiers tested are the cheap, honest class (token-confidence signals; a CV'd linear/logistic
  verifier). Heavier methods — large-K sampling, process/step-level verifiers, a fully trained reward
  model — were **not** tested. The negative result is specific to this regime and budget; it does not
  claim no verifier can ever win, only that the cheap, honest verifier space here is exhausted.

## Reproduce

```bash
cd experiments/keystone
pip install -r requirements.txt
# one GPU sweep (~40 min on a T4)
python gen_candidates.py --n_questions 150 --k 6 --out candidates.jsonl
# the rest is CPU, seconds each
python score_energy.py --in candidates.jsonl --scorer trained --features both --out scored_both.jsonl
python analyze.py --in scored_both.jsonl          # Finding 1
python efficiency.py --in candidates.jsonl        # Finding 2
```