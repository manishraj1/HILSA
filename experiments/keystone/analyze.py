#!/usr/bin/env python3
"""
Keystone analysis (v2) — model-agnostic, decisive.

Reads scored candidates (jsonl) and reports selection accuracy for:
  - avg single-sample (random pick)
  - majority vote (self-consistency)            <- THE BAR TO BEAT
  - verifier-best (argmin energy)
  - confidence-weighted self-consistency (CW-SC) <- the new contender
  - oracle pass@K (ceiling)
plus energy AUC and a PAIRED bootstrap test of each contender vs majority vote.

Required fields per row: qid, is_correct, energy, pred
(energy convention: LOWER = better)

Usage:
  python analyze.py --in scored_logprob.jsonl
  python analyze.py --in scored_probe.jsonl --tau 1.0
"""
import argparse, json, sys
from collections import defaultdict, Counter
import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        sys.exit("No rows found.")
    return rows


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cwsc_pick(preds, energies, gold, tau, lower_better):
    """Confidence-weighted self-consistency: weight each answer by softmax of its
    (standardized) verifier score, sum per distinct answer, pick the heaviest.
    tau->0 approaches verifier-best; tau->inf approaches plain majority."""
    s = np.array([(-e if lower_better else e) for e in energies], float)
    sd = s.std()
    if sd < 1e-9:
        w = np.ones(len(s)) / len(s)
    else:
        z = (s - s.mean()) / sd / max(tau, 1e-6)
        z -= z.max()
        w = np.exp(z); w /= w.sum()
    agg = defaultdict(float)
    for p, wi in zip(preds, w):
        agg[str(p)] += wi
    best = max(agg, key=agg.get)
    return gold is not None and best == gold


def per_question(rows, lower_better, tau):
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    Q = {}
    for qid, rs in by_q.items():
        flags = [bool(r["is_correct"]) for r in rs]
        energies = [float(r["energy"]) for r in rs]
        preds = [str(r.get("pred")) for r in rs]
        gold = next((p for p, c in zip(preds, flags) if c), None)
        idx = int(np.argmin(energies)) if lower_better else int(np.argmax(energies))
        mode_pred = Counter(preds).most_common(1)[0][0]
        Q[qid] = dict(
            rand=float(np.mean(flags)),
            majority=(gold is not None and mode_pred == gold),
            vbest=flags[idx],
            cwsc=cwsc_pick(preds, energies, gold, tau, lower_better),
            oracle=any(flags),
        )
    return Q


def boot(vals, B, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(vals, float)
    bs = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def boot_diff(a, b, B, seed=0):
    """Paired bootstrap of (a - b) over questions. Returns mean diff + CI."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pct(m, lo, hi):
    return f"{m*100:5.1f}%  [{lo*100:4.1f}, {hi*100:4.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--energy_lower_is_better", default="true")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    lower = args.energy_lower_is_better.lower() in ("true", "1", "yes")

    rows = load(args.inp)
    Q = per_question(rows, lower, args.tau)
    qids = list(Q)
    n_q = len(qids)
    print(f"\nLoaded {len(rows)} candidates | {n_q} questions | "
          f"K~{len(rows)/n_q:.1f} | tau={args.tau}\n")

    energies = [float(r["energy"]) for r in rows]
    labels = [bool(r["is_correct"]) for r in rows]
    score = [-e for e in energies] if lower else energies
    a = auc(score, labels)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    rng = np.random.default_rng(1); aucs = []
    for _ in range(args.boot):
        samp = rng.choice(qids, size=n_q, replace=True)
        s, l = [], []
        for q in samp:
            for r in by_q[q]:
                s.append(-float(r["energy"]) if lower else float(r["energy"]))
                l.append(bool(r["is_correct"]))
        aucs.append(auc(s, l))
    a_lo, a_hi = np.nanpercentile(aucs, 2.5), np.nanpercentile(aucs, 97.5)

    rnd = boot([Q[q]["rand"] for q in qids], args.boot)
    maj = boot([Q[q]["majority"] for q in qids], args.boot)
    vb = boot([Q[q]["vbest"] for q in qids], args.boot)
    cw = boot([Q[q]["cwsc"] for q in qids], args.boot)
    orc = boot([Q[q]["oracle"] for q in qids], args.boot)

    print("  accuracy by selection strategy (95% bootstrap CI over questions)")
    print("  " + "-" * 58)
    print(f"  avg single-sample (random pick)  : {pct(*rnd)}")
    print(f"  majority vote (self-consistency) : {pct(*maj)}   <- bar")
    print(f"  verifier-best (argmin energy)    : {pct(*vb)}")
    print(f"  CW self-consistency (tau={args.tau:<4})    : {pct(*cw)}")
    print(f"  oracle pass@K (ceiling)          : {pct(*orc)}")
    print(f"\n  energy AUC (0.5 = no signal)     : {a:5.3f}  [{a_lo:.3f}, {a_hi:.3f}]")

    maj_v = [Q[q]["majority"] for q in qids]
    dvb = boot_diff([Q[q]["vbest"] for q in qids], maj_v, args.boot)
    dcw = boot_diff([Q[q]["cwsc"] for q in qids], maj_v, args.boot)
    print("\n  vs majority vote (paired bootstrap of the difference):")
    print(f"    verifier-best - majority : {dvb[0]*100:+5.1f}pp  [{dvb[1]*100:+.1f}, {dvb[2]*100:+.1f}]")
    print(f"    CW-SC        - majority : {dcw[0]*100:+5.1f}pp  [{dcw[1]*100:+.1f}, {dcw[2]*100:+.1f}]")

    sweep = []
    for t in (0.2, 0.5, 1.0, 2.0, 5.0):
        acc = np.mean([cwsc_pick([str(r.get("pred")) for r in by_q[q]],
                                 [float(r["energy"]) for r in by_q[q]],
                                 next((str(r.get("pred")) for r in by_q[q] if r["is_correct"]), None),
                                 t, lower) for q in qids])
        sweep.append((t, acc))
    print("\n  CW-SC tau sweep (EXPLORATORY -- tuned on this set, treat as upper bound):")
    print("    " + "  ".join(f"t={t}:{acc*100:.1f}%" for t, acc in sweep))

    signal = a_lo > 0.5
    cw_beats_pt = cw[0] >= maj[0]
    cw_beats_sig = dcw[1] > 0
    print("\n  VERDICT:")
    if not signal:
        print("  >> NO SIGNAL: AUC CI includes 0.5. The verifier doesn't track correctness.")
    elif cw_beats_sig:
        print(f"  >> BEATS SELF-CONSISTENCY: CW-SC > majority by {dcw[0]*100:+.1f}pp, CI excludes 0.")
        print("     Real win. Lock it in, then push the verifier further.")
    elif cw_beats_pt:
        print(f"  >> TIED / EDGES AHEAD: signal is real (AUC {a:.2f}); CW-SC matches or")
        print("     slightly beats majority but CI crosses 0. Promising -- needs a stronger")
        print("     verifier or more questions to separate. Scale n and/or improve the signal.")
    else:
        print(f"  >> REAL SIGNAL, BELOW THE BAR: verifier works (AUC {a:.2f}) but majority")
        print("     vote still wins. The trick to beat is self-consistency, not random.")
        print("     Next: a stronger free signal (min/last-span logprob) or a trained verifier.")
    print()


if __name__ == "__main__":
    main()