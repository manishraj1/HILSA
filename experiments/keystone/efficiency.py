#!/usr/bin/env python3
"""
Door 1 — the efficiency keystone (verifier-free).

Question: can we reach majority@K accuracy using FEWER than K samples on average,
by stopping early once the samples agree? No verifier, no training -- just counting.

Reads candidates.jsonl (fields: qid, pred, gold; K samples per question).
Two analyses, both offline / CPU-only:

  STATIC saturation : majority-vote accuracy using only the first m samples
                      (m=1..K), averaged over many random orderings.
  ADAPTIVE stopping : draw one sample at a time, stop when the leading answer's
                      margin over the runner-up reaches L; report avg samples used
                      and accuracy, for L = 1,2,3.

Verdict (pre-registered): an EFFICIENCY WIN is any method that matches majority@K
accuracy (paired-bootstrap difference's CI includes 0) at < K average samples.
If majority@(K-2) is clearly below @K AND nothing matches @K below ~5.5 avg
samples, the honest finding is "this task needs the samples."

Usage:
  python efficiency.py --in candidates.jsonl
"""
import argparse, json
from collections import defaultdict, Counter
import numpy as np


def load(path):
    by_q = defaultdict(list)
    gold = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_q[r["qid"]].append(str(r.get("pred")))
            gold[r["qid"]] = str(r.get("gold"))
    return by_q, gold


def majority(preds):
    """Most common answer; ties broken by first appearance (caller shuffles)."""
    return Counter(preds).most_common(1)[0][0]


def margin_stop(preds, L, K):
    """Walk preds in given order, stop when (top1 - top2) >= L or at K. Return
    (samples_used, chosen_answer)."""
    counts = Counter()
    for i, p in enumerate(preds, start=1):
        counts[p] += 1
        top = counts.most_common(2)
        lead = top[0][1] - (top[1][1] if len(top) > 1 else 0)
        if lead >= L or i == K:
            return i, top[0][0]
    return len(preds), majority(preds)


def per_question_metrics(by_q, gold, K, Ls, R, seed=0):
    rng = np.random.default_rng(seed)
    qids = list(by_q)
    static = {m: [] for m in range(1, K + 1)}       # qid-level expected acc
    adapt_acc = {L: [] for L in Ls}
    adapt_n = {L: [] for L in Ls}
    for q in qids:
        preds = by_q[q]; g = gold[q]; k = len(preds)
        s_hits = {m: 0.0 for m in range(1, K + 1)}
        a_hits = {L: 0.0 for L in Ls}; a_n = {L: 0.0 for L in Ls}
        for _ in range(R):
            order = list(rng.permutation(k))
            seq = [preds[i] for i in order]
            for m in range(1, K + 1):
                s_hits[m] += (majority(seq[:m]) == g)
            for L in Ls:
                n, ans = margin_stop(seq, L, K)
                a_hits[L] += (ans == g); a_n[L] += n
        for m in range(1, K + 1):
            static[m].append(s_hits[m] / R)
        for L in Ls:
            adapt_acc[L].append(a_hits[L] / R)
            adapt_n[L].append(a_n[L] / R)
    return qids, static, adapt_acc, adapt_n


def boot(vals, B=4000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(vals, float)
    bs = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def boot_diff(a, b, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--margins", default="1,2,3")
    args = ap.parse_args()
    Ls = [int(x) for x in args.margins.split(",")]

    by_q, gold = load(args.inp)
    K = max(len(v) for v in by_q.values())
    qids, static, adapt_acc, adapt_n = per_question_metrics(by_q, gold, K, Ls, args.R)
    n_q = len(qids)
    print(f"\nLoaded {n_q} questions | K={K} | {args.R} orderings/question\n")

    maj_full = static[K]                       # majority@K per-question acc
    mf = boot(maj_full)

    print("  STATIC saturation — majority vote on first m samples")
    print("  " + "-" * 52)
    for m in range(1, K + 1):
        acc = boot(static[m])
        tag = "  <- majority@K (the target)" if m == K else ""
        print(f"   m={m} (avg samples {m:.1f}) : {acc[0]*100:5.1f}%  "
              f"[{acc[1]*100:4.1f}, {acc[2]*100:4.1f}]{tag}")

    print("\n  ADAPTIVE early-stopping — stop when lead reaches margin L")
    print("  " + "-" * 52)
    rows = []
    for L in Ls:
        acc = boot(adapt_acc[L]); avg_n = float(np.mean(adapt_n[L]))
        d = boot_diff(adapt_acc[L], maj_full)
        matches = (d[1] <= 0 <= d[2]) or d[0] > 0      # indistinguishable from @K, or better
        rows.append((f"adaptive L={L}", avg_n, acc, d, matches))
        flag = "matches @K" if matches else "below @K"
        print(f"   L={L}: avg samples {avg_n:4.2f}  acc {acc[0]*100:5.1f}%  "
              f"[{acc[1]*100:4.1f}, {acc[2]*100:4.1f}]  "
              f"(vs @K: {d[0]*100:+.1f}pp [{d[1]*100:+.1f},{d[2]*100:+.1f}] -> {flag})")

    # also test fixed@(K-2) vs @K (the kill condition's specific clause)
    if K - 2 >= 1:
        dk2 = boot_diff(static[K - 2], maj_full)
        k2_matches = (dk2[1] <= 0 <= dk2[2]) or dk2[0] > 0
        rows.append((f"fixed K={K-2}", float(K - 2), boot(static[K - 2]), dk2, k2_matches))

    # ---- verdict ----
    winners = [(name, n, acc, d) for (name, n, acc, d, m) in rows if m and n < K]
    print("\n  VERDICT (pre-registered):")
    if winners:
        name, n, acc, d = min(winners, key=lambda x: x[1])
        saved = (K - n) / K * 100
        if n <= 5.5:
            print(f"  >> EFFICIENCY WIN: {name} reaches {acc[0]*100:.1f}% "
                  f"(indistinguishable from majority@{K} = {mf[0]*100:.1f}%)")
            print(f"     at {n:.2f} avg samples -> {saved:.0f}% less compute for the same accuracy.")
            print("     This is the energy-saving result. Lock it in.")
        else:
            print(f"  >> MARGINAL: best match is {name} at {n:.2f} avg samples "
                  f"(> 5.5). Savings exist but are small.")
    else:
        print(f"  >> NO CHEAP SAVINGS: nothing matches majority@{K} ({mf[0]*100:.1f}%) "
              f"below {K} samples.")
        print("     Honest finding: this task needs the samples. Record it and move on.")
    print()


if __name__ == "__main__":
    main()