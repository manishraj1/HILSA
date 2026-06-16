#!/usr/bin/env python3
"""
Keystone step 2 — scoring. Writes an `energy` field analyze.py reads
(convention: LOWER energy = better candidate).

Single free signals (no training), all derived from the answer's token log-probs:
  --scorer logprob    energy = -mean_logprob        (baseline; washes out broken steps)
  --scorer minlogprob energy = -min_logprob         (the single most-broken token)
  --scorer bottom2    energy = -bottom2_mean_logprob (the two weakest links)
  --scorer lastspan   energy = -last_k_mean_logprob (confidence on the conclusion)

Trained verifiers (cross-validated by QUESTION so no leakage), reward-model style:
  --scorer trained    logistic regression over the scalar confidence features
                      (optionally + hidden states).  energy = -P(correct)
  --scorer probe      logistic regression over hidden states only (legacy)

The decisive read is always CW-SC vs majority in analyze.py, fed each scorer's output.

Usage:
  python score_energy.py --in candidates.jsonl --scorer minlogprob --out scored_min.jsonl
  python score_energy.py --in candidates.jsonl --scorer trained --features scalar --out scored_trained.jsonl
  python score_energy.py --in candidates.jsonl --hidden hidden.npy --scorer trained --features both --out scored_both.jsonl
"""
import argparse, json
import numpy as np

SCALAR_KEYS = ["mean_logprob", "min_logprob", "bottom2_mean_logprob",
               "last_k_mean_logprob", "last_logprob", "std_logprob", "n_tokens"]

SINGLE = {  # scorer name -> (feature key)
    "logprob": "mean_logprob",
    "minlogprob": "min_logprob",
    "bottom2": "bottom2_mean_logprob",
    "lastspan": "last_k_mean_logprob",
}


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_single(rows, key):
    for r in rows:
        v = r.get(key)
        r["energy"] = float("nan") if v is None else -float(v)  # higher logprob -> lower energy
    return rows


def _aligned(rows):
    if all("row_idx" in r for r in rows):
        return sorted(rows, key=lambda r: r["row_idx"])
    return rows


def score_trained(rows, hidden_path, folds, features):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GroupKFold
    except ImportError:
        raise SystemExit("trained/probe scorer needs scikit-learn:  pip install scikit-learn")

    rows = _aligned(rows)
    parts = []
    if features in ("scalar", "both"):
        parts.append(np.array([[float(r.get(k, 0.0)) for k in SCALAR_KEYS] for r in rows], float))
    if features in ("hidden", "both"):
        H = np.load(hidden_path).astype(np.float32)
        assert len(H) == len(rows), f"hidden rows {len(H)} != jsonl rows {len(rows)}"
        parts.append(H)
    if not parts:
        raise SystemExit("--features must be scalar, hidden, or both")
    X = np.concatenate(parts, axis=1)

    y = np.array([1 if r["is_correct"] else 0 for r in rows])
    groups = np.array([r["qid"] for r in rows])
    oof = np.full(len(rows), np.nan)
    gkf = GroupKFold(n_splits=min(folds, len(set(groups))))
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = float(y[tr].mean()); continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    for r, p in zip(rows, oof):
        r["energy"] = -float(p)            # higher P(correct) -> lower energy
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--scorer", required=True,
                    choices=list(SINGLE) + ["trained", "probe"])
    ap.add_argument("--features", default="scalar", choices=["scalar", "hidden", "both"])
    ap.add_argument("--hidden", default="hidden.npy")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="scored.jsonl")
    args = ap.parse_args()

    rows = load_rows(args.inp)
    if args.scorer in SINGLE:
        rows = score_single(rows, SINGLE[args.scorer])
    elif args.scorer == "probe":
        rows = score_trained(rows, args.hidden, args.folds, "hidden")
    else:  # trained
        rows = score_trained(rows, args.hidden, args.folds, args.features)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Scored {len(rows)} rows with '{args.scorer}' "
          f"(features={args.features if args.scorer=='trained' else 'n/a'}) -> {args.out}")
    print(f"Next:  python analyze.py --in {args.out}")


if __name__ == "__main__":
    main()