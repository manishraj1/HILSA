# Inference-Time Verification Does Not Beat Self-Consistency on a 1.5B Model, but Adaptive Stopping Recovers Its Accuracy at ~⅓ Less Compute

*A reproducible empirical study. Independent work.*

## Abstract

We study inference-time answer selection on a sub-2B model (Qwen2.5-1.5B-Instruct)
on GSM8K. We ask two questions. First, can an inference-time verifier select a better
answer from K sampled chains than majority vote? Across six verifiers — free token-confidence
signals (mean, min, bottom-2, and conclusion-span log-probability) and cross-validated trained
verifiers over scalar and hidden-state features — the answer is no: none beats majority vote,
and the verifiers carry real but insufficient signal (AUC 0.71–0.74). Second, can we reach
majority-vote accuracy with fewer samples by stopping once the samples agree? Yes: a simple
agreement-margin stopping rule matches majority@6 accuracy using 4.12 samples on average — a 31%
compute reduction at indistinguishable accuracy. We position both results against existing work:
the efficiency result reproduces, on a small model, the known behaviour of adaptive self-consistency
(Aggarwal et al., 2023) and compute-optimal scaling (Snell et al., 2024); the verifier result is a
clean negative result for cheap inference-time verification at this scale. We do not propose a new
method. We contribute a careful, fully reproducible measurement, including the negative result, with
all data and code released.

## 1. Introduction

Self-consistency (Wang et al., 2022) samples multiple chains of thought and returns the most
frequent answer, reliably improving reasoning accuracy at the cost of K× inference. Two natural
questions follow: can we *select better* than the majority within the K samples, and can we *spend
less* than K to get the same accuracy? Both are active research areas, mostly studied on models of
7B parameters and above. This report asks both on a 1.5B model, under strict pre-registration: every
claim, baseline, and kill condition was fixed before results were read, and every number carries a
bootstrap confidence interval.

Our motivation is honesty and reproducibility rather than novelty. The contribution is a clean
measurement on a small model, including a negative result that the literature's positive framing can
obscure: at this scale, cheap inference-time verifiers do not beat counting.

## 2. Related work

**Self-consistency and selection.** Wang et al. (2022) introduced majority voting over sampled
reasoning paths. A large subsequent literature improves *which* answer is selected using verifiers
and reward models, and *how many* samples are drawn.

**Adaptive sampling / early stopping.** Aggarwal et al. (2023, "Adaptive-Consistency", arXiv:2305.11860)
stop sampling per question once interim agreement is high, reducing samples up to 7.9× with <0.1%
accuracy loss. ESC (Li et al., 2024) stops on windows of agreeing samples. RASC (Wan et al., 2025,
arXiv:2408.17017) and ReASC (Kim et al., 2026, arXiv:2601.02970) make stopping *reliability-aware*,
weighting samples by confidence rather than counting them equally. Probing hidden states to predict
correctness has also been studied directly (Zhang et al., 2025). Our adaptive-stopping experiment
is a simple instance of this family, and our result is consistent with theirs on a smaller model.

**Compute-optimal / difficulty-aware allocation.** Snell et al. (2024, arXiv:2408.03314) allocate
test-time compute per prompt by estimated difficulty, improving efficiency ~4× and showing a small
model plus test-time compute can outperform a much larger one. OSCA (Zhang et al., 2024,
arXiv:2410.22480), CODA (2026), and DORA (2025) study related allocation problems. These cover the
"spend more on hard questions, less on easy ones" idea at a level of generality beyond this report.

**Positioning.** Against this body of work, we claim no new method. We provide a reproduction on a
sub-2B model and a clean negative result on cheap verifier-based selection at that scale.

## 3. Setup

Qwen2.5-1.5B-Instruct; GSM8K test; 150 questions; K=6 samples each (one greedy, five at temperature
0.8, top-p 0.95); max 400 new tokens; fixed 4-shot CoT prompt; seed 0; 900 candidates total. Single
T4 GPU. Correctness is exact match on the final integer. Verifiers are scored offline. All
confidence intervals are 95% bootstrap over questions; verifier-vs-baseline comparisons use a paired
bootstrap of the per-question difference. Trained verifiers are evaluated with GroupKFold by question
(no leakage). Data and code are released.

## 4. Findings

### 4.1 Verifier selection does not beat self-consistency (negative)

Baselines: random pick 50.2% [44.3, 56.0]; majority vote 68.7% [60.7, 76.0]; oracle pass@K 83.3%
[77.3, 89.3]. The entire selectable margin is the 14.6 points between majority and oracle.

Six verifiers, best selection strategy (confidence-weighted self-consistency) vs majority vote
(paired difference):

| verifier | AUC | CW-SC − majority |
|---|---|---|
| mean log-prob | 0.71 | −4.7pp [−8.7, −0.7] |
| min log-prob | 0.66 | −8.0pp [−12.7, −4.0] |
| bottom-2 log-prob | 0.68 | −8.7pp [−13.3, −4.7] |
| conclusion-span log-prob | ~0.50 | −9.3pp [−14.7, −4.0] |
| trained (scalar, CV) | 0.74 | −6.0pp [−10.7, −2.0] |
| trained (scalar+hidden, CV) | 0.73 | −2.0pp [−8.0, +3.3] |

No verifier beats majority vote; most are reliably behind. The "sharper" single signals (min,
bottom-2, span) did worse, falsifying the hypothesis that one low-probability token reliably flags a
wrong answer. The convergence of six independent methods to a band below the baseline indicates a
structural ceiling: when a correct answer exists in the sample set it is usually already the
plurality, so the cases a verifier could win are exactly those the model produced least confidently.

### 4.2 Adaptive early-stopping recovers full accuracy cheaply (positive, reproduction)

Stopping when the leading answer's margin over the runner-up reaches 2 (cap 6) reaches 66.7%
[59.3, 73.8], indistinguishable from majority@6's 66.8% (paired −0.1pp [−0.3, +0.1]), at 4.12
average samples — a 31% compute reduction. Easy questions stop in 2–3 samples; hard ones spend the
full budget. This reproduces, on a 1.5B model, the efficiency of adaptive self-consistency (Aggarwal
et al., 2023).

## 5. Discussion

On a small model on grade-school math, the useful lever is not a smarter judge of answers but
knowing when to stop asking. Cheap inference-time verification — including a trained verifier over
hidden states — does not beat majority vote for selection at this scale, while agreement-based
stopping captures the efficiency that the answer-selection literature also reports.

## 6. Limitations

One model, one benchmark, arithmetic only; not claimed to generalize. K ≤ 6, so we study stopping
*earlier* than 6, not whether more than 6 helps. Verifiers tested are the cheap, honest class;
heavier methods (large-K sampling, process/step verifiers, fully trained reward models) were not
tested — the negative result is specific to this regime and budget and does not claim no verifier can
ever win.

## 7. Reproducibility

All candidates, scores, scripts, and the pre-registration (`SPEC.md`) are in `experiments/keystone/`.
Re-run: `gen_candidates.py` (one GPU sweep) → `score_energy.py` → `analyze.py` (Finding 1) →
`efficiency.py` (Finding 2).

## References

- Wang et al., 2023 (ICLR). Self-Consistency Improves Chain-of-Thought Reasoning in Language Models. arXiv:2203.11171.
- Aggarwal et al., 2023. Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning. arXiv:2305.11860.
- Li et al., 2024 (ICLR). Escape Sky-high Cost: Early-Stopping Self-Consistency for Multi-step Reasoning. arXiv:2401.10480.
- Wan et al., 2025 (NAACL). Reasoning-Aware Self-Consistency. arXiv:2408.17017.
- Kim et al., 2026. Reliability-Aware Adaptive Self-Consistency. arXiv:2601.02970.
- Snell et al., 2024. Scaling LLM Test-Time Compute Optimally. arXiv:2408.03314.
- Zhang et al., 2024. Scaling LLM Inference with Optimized Sample Compute Allocation (OSCA). arXiv:2410.22480.
- Zhang et al., 2025. Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification. arXiv:2504.05419.