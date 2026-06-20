# Inference-Time Verification vs. Adaptive Compute on a Small Model

A small, fully reproducible study on **Qwen2.5-1.5B-Instruct** / **GSM8K**, asking two
questions about getting more out of sampled chain-of-thought reasoning. Every claim is
pre-registered, every number carries a bootstrap confidence interval, and all data and
code are included so the results can be re-run end to end.

| Status | Link |
| :--- | :--- |
| **Journal Track** | 📥 Under Review at **TMLR** (Transactions on Machine Learning Research) |
| **Preprint & Citation** | 📑 [ResearchGate Publication](https://www.researchgate.net/publication/407307300_The_Verification_Ceiling_and_the_Gating_Frontier_Inference-Time_Selection_and_Adaptive_Compute_on_a_15B_Model) |
| **Full Report** | 📝 [`experiments/keystone/REPORT.md`](experiments/keystone/REPORT.md) |

## Summary

1. **Verifier selection does not beat self-consistency (negative result).** Across six
   inference-time verifiers — free token-confidence signals (mean / min / bottom-2 /
   conclusion-span log-probability) and cross-validated trained verifiers over scalar and
   hidden-state features — none improves answer selection over plain majority vote. The
   verifiers carry real but insufficient signal (AUC 0.71–0.74). This is consistent with
   the original self-consistency paper, where log-probability weighting was already found
   to roughly match majority vote.

2. **Adaptive early-stopping recovers full accuracy at ~⅓ less compute (positive result).**
   Stopping sampling once the answers agree (margin ≥ 2) matches majority-vote accuracy
   using **4.12 samples instead of 6** — a ~31% compute reduction at statistically
   indistinguishable accuracy, with no verifier and no training. This reproduces, on a
   1.5B model, the behaviour of adaptive self-consistency (Aggarwal et al., 2023; Li et
   al., 2024).

| strategy | accuracy |
|---|---|
| random single sample | 50.2% [44.3, 56.0] |
| majority vote (self-consistency) | 68.7% [60.7, 76.0] |
| best verifier (selection) | ≤ majority vote (all six below) |
| oracle pass@K (ceiling) | 83.3% [77.3, 89.3] |
| adaptive stopping (margin 2) | matches majority vote at 4.12 avg samples |

The takeaway: on a small model on grade-school math, the useful lever is not a smarter
judge of answers but knowing when to stop sampling.

## What's here

```
experiments/keystone/
  gen_candidates.py    generation (Qwen2.5-1.5B on GSM8K; the one GPU step)
  score_energy.py      verifiers: log-prob signals + cross-validated trained verifier
  analyze.py           Finding 1 — verifier vs majority vote, with bootstrap CIs
  efficiency.py        Finding 2 — saturation curve + adaptive-stopping frontier
  SPEC.md              pre-registration (claims and kill conditions, fixed in advance)
  RESULTS.md           findings
  REPORT.md            full writeup with related work and references
  requirements.txt
  candidates.jsonl     committed data (so the numbers reproduce without a rerun)
  scored_*.jsonl       per-verifier scores
```

## Reproduce

```bash
cd experiments/keystone
pip install -r requirements.txt
python gen_candidates.py --n_questions 150 --k 6 --out candidates.jsonl   # ~40 min on a T4
python score_energy.py --in candidates.jsonl --scorer trained --features both --out scored_both.jsonl
python analyze.py --in scored_both.jsonl     # Finding 1
python efficiency.py --in candidates.jsonl   # Finding 2
```

The scoring and analysis steps are CPU-only and run in seconds on the committed data.

## Scope

One model, one benchmark, arithmetic reasoning, K ≤ 6. Not claimed to generalize. The
verifiers tested are the cheap, honest class; heavier methods (large-K sampling,
process/step verifiers, fully trained reward models) were not tested. The negative result
is specific to this regime and does not claim no verifier can ever win. See
[`REPORT.md`](experiments/keystone/REPORT.md) for full limitations and citations.
##
**Empirical evaluation and codebase produced by Manish Raj Vangari.**
