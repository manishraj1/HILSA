#!/usr/bin/env python3
"""
Keystone step 1 — candidate generation (Qwen2.5-1.5B-Instruct on GSM8K).

For each question: sample K answers (sample 0 = greedy, rest = temperature),
exact-match against gold, and record TWO honest, model-derived signals per answer:
  - mean_logprob : true under-model mean token log-probability (temperature-free,
                   from a teacher-forced forward pass) -> free baseline verifier
  - hidden state : mean of the LAST-layer hidden states over the answer tokens
                   -> saved to hidden.npy for a trained linear probe later

Clean room: imports nothing from zero_defect_shell / ruliad_engine / mfu_v2 / stage9.

Outputs:
  candidates.jsonl  rows: {qid, sample_idx, pred, gold, is_correct, tokens,
                           mean_logprob, row_idx}
  hidden.npy        float16 array (n_rows, hidden_dim), row order == jsonl row_idx

Usage (Colab T4):
  python gen_candidates.py --n_questions 150 --k 6 --out candidates.jsonl
"""
import argparse, json, re, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = ("You are a careful math tutor. Solve the problem step by step, "
          "then end with a line exactly of the form '#### <final integer answer>'.")

# 4-shot exemplars (canonical GSM8K-style)
FEWSHOT = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?",
     "In April she sold 48 clips. In May she sold 48 / 2 = 24 clips. "
     "Altogether 48 + 24 = 72.\n#### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
     "of babysitting. How much did she earn?",
     "Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned "
     "50 * 0.2 = $10.\n#### 10"),
    ("Betty is saving for a $100 wallet. She has half the money she needs. Her "
     "parents give her $15, and her grandparents give twice as much as her "
     "parents. How much more money does Betty need?",
     "Half of 100 is 50. Grandparents give 2 * 15 = 30. She now has "
     "50 + 15 + 30 = 95. She still needs 100 - 95 = 5.\n#### 5"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many "
     "pages does he write a year?",
     "Each time he writes 3 * 2 = 6 pages. Twice a week is 6 * 2 = 12 pages. "
     "Over a year that is 12 * 52 = 624.\n#### 624"),
]


def build_prompt_ids(tok, question, device):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in FEWSHOT:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": question})
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    # apply_chat_template may return a Tensor or a BatchEncoding depending on version;
    # normalize to a plain 2-D LongTensor so downstream code never guesses.
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    return enc.to(device)


def parse_answer(text):
    m = re.findall(r"####\s*(-?[\d,]+)", text)
    if m:
        return m[-1].replace(",", "").strip()
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else None


def gold_answer(ans_field):
    return ans_field.split("####")[-1].replace(",", "").strip()


def logprob_features(tok_lps, last_k=8):
    """Rich confidence features from per-token log-probs of the generated answer."""
    if not tok_lps:
        nan = float("nan")
        return dict(mean_logprob=nan, min_logprob=nan, bottom2_mean_logprob=nan,
                    last_k_mean_logprob=nan, last_logprob=nan, std_logprob=nan, n_tokens=0)
    a = np.asarray(tok_lps, float)
    s = np.sort(a)
    return dict(
        mean_logprob=float(a.mean()),
        min_logprob=float(a.min()),                       # single most-broken token
        bottom2_mean_logprob=float(s[:2].mean()),         # the weakest links
        last_k_mean_logprob=float(a[-last_k:].mean()),    # conclusion span
        last_logprob=float(a[-1]),
        std_logprob=float(a.std()),
        n_tokens=int(len(a)),
    )


@torch.no_grad()
def score_sequence(model, full_seq, prompt_len, eos_id):
    """Trim at first EOS, then one forward pass -> (features_dict, hidden_vec)."""
    cont = full_seq[prompt_len:]
    eos = (cont == eos_id).nonzero()
    end = prompt_len + int(eos[0]) + 1 if len(eos) > 0 else len(full_seq)
    seq = full_seq[:end].unsqueeze(0)
    out = model(seq, output_hidden_states=True)
    logits = out.logits[0].float()
    last_hidden = out.hidden_states[-1][0]
    lp = torch.log_softmax(logits, dim=-1)
    tok_lps = [lp[pos - 1, full_seq[pos]].item() for pos in range(prompt_len, end)]
    feats = logprob_features(tok_lps)
    hvec = last_hidden[prompt_len:end].mean(dim=0).float().cpu().numpy()
    return feats, hvec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_questions", type=int, default=150)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="candidates.jsonl")
    ap.add_argument("--hidden_out", default="hidden.npy")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {MODEL}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
    model.eval()
    eos_id = tok.eos_token_id

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n_questions))

    f = open(args.out, "w")
    hidden_rows, row_idx = [], 0
    for qi, ex in enumerate(ds):
        qid = f"q{qi:04d}"
        gold = gold_answer(ex["answer"])
        prompt = build_prompt_ids(tok, ex["question"], device)
        plen = prompt.shape[1]
        attn = torch.ones_like(prompt)
        seqs = []
        # sample 0: greedy
        g = model.generate(prompt, attention_mask=attn, do_sample=False,
                           max_new_tokens=args.max_new_tokens, pad_token_id=eos_id)
        seqs.append(g[0])
        # samples 1..K-1: temperature, batched
        if args.k > 1:
            torch.manual_seed(args.seed * 1000 + qi)
            s = model.generate(prompt, attention_mask=attn, do_sample=True,
                              temperature=args.temperature, top_p=args.top_p,
                              num_return_sequences=args.k - 1,
                              max_new_tokens=args.max_new_tokens, pad_token_id=eos_id)
            seqs.extend([s[j] for j in range(s.shape[0])])

        for si, full in enumerate(seqs):
            feats, hvec = score_sequence(model, full, plen, eos_id)
            text = tok.decode(full[plen:], skip_special_tokens=True)
            pred = parse_answer(text)
            rec = dict(qid=qid, sample_idx=si, pred=pred, gold=gold,
                       is_correct=bool(pred is not None and pred == gold),
                       tokens=feats["n_tokens"], row_idx=row_idx, **feats)
            f.write(json.dumps(rec) + "\n")
            hidden_rows.append(hvec.astype(np.float16))
            row_idx += 1
        if (qi + 1) % 10 == 0:
            f.flush()
            print(f"  {qi+1}/{args.n_questions} questions | {row_idx} candidates")
            torch.cuda.empty_cache()

    f.close()
    np.save(args.hidden_out, np.stack(hidden_rows))
    print(f"\nWrote {row_idx} candidates -> {args.out}")
    print(f"Saved hidden states {np.stack(hidden_rows).shape} -> {args.hidden_out}")


if __name__ == "__main__":
    main()