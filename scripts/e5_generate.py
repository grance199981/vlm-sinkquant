"""E5 stage 0: BF16 reference thinking chains (greedy) for Qwen3-VL-Thinking on V* or MathVista.

The prompt is the model's own chat template (ends with '<think>\\n').
  V*        : the question with the answer line of 2610.15855 ('Put only the letter ... inside \\boxed{}')
  MathVista : the official 'query' field (hint + question + choices) plus 'Put the final answer inside \\boxed{}.'
Greedy decoding (the generation_config default is sampling; greedy keeps the reference
reproducible), max_new_tokens = 4096. With --batch-size > 1 prompts are left-padded; each item's
unpadded prompt ids are checked against the single-item prompt that stage 1 rebuilds. Per item we
store the prompt ids hash, the generated ids (up to and including the first EOS), the position of
'</think>', the extracted answer and whether the cap was hit. MathVista scoring is rule-based
(\\boxed content vs. answer; letters map to choices; numbers compared at the given precision) and
only approximates the official GPT-based extraction.

Usage: python e5_generate.py --model-dir M --model-tag T --dataset {vstar,mathvista} --data-root D
           --out-dir OUT [--batch-size B] [--limit N] [--shard-index i --num-shards n] [--skip-done f1,f2]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_sink_stats import pick_vstar  # noqa: E402

MV_FORMAT = r"Put the final answer inside \boxed{}."


def pick_mathvista(root, n):
    out = []
    for line in open(os.path.join(root, "subset", "questions.jsonl")):
        q = json.loads(line)
        q.update(file=os.path.join("subset", q["file"]), label=q["answer"], raw_question=q["question"],
                 question=q["query"].rstrip() + "\n" + MV_FORMAT)
        out.append(q)
    return out[:n]


def pick(dataset, root):
    return pick_vstar(root, 10_000) if dataset == "vstar" else pick_mathvista(root, 10_000)


def prompt_text(proc, img, question):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert text.rstrip().endswith("<think>"), text[-80:]
    return text


def build_prompt(proc, img, question):
    return proc(text=[prompt_text(proc, img, question)], images=[img], return_tensors="pt").to("cuda")


def ids_hash(ids):
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def boxed(text):
    m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return m[-1].strip() if m else None


def score(it, ans_text, dataset):
    if dataset == "vstar":
        m = re.findall(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", ans_text)
        pred = m[-1] if m else None
        return pred, pred == it["label"]
    pred = boxed(ans_text)
    if pred is None:
        return None, False
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().strip(".").lower()
    if it["question_type"] == "multi_choice":
        p = pred.strip("() ")
        if len(p) == 1 and p.isalpha() and ord(p.upper()) - 65 < len(it["choices"]):
            p = it["choices"][ord(p.upper()) - 65]
        return pred, norm(p) == norm(it["answer"])
    if it["answer_type"] in ("integer", "float"):
        try:
            x = float(re.sub(r"[^0-9.\-eE]", "", pred.replace("\\frac", "")))
            y = float(it["answer"])
        except ValueError:
            return pred, False
        if it["answer_type"] == "integer":
            return pred, round(x) == round(y) and abs(x - round(x)) < 1e-6
        prec = int(it["precision"] or 2)
        return pred, round(x, prec) == round(y, prec)
    return pred, norm(pred) == norm(it["answer"])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--dataset", choices=["vstar", "mathvista"], default="vstar")
    ap.add_argument("--data-root", "--vstar-root", dest="data_root", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--sample", action="store_true",
                    help="sample with the model's generation_config (temperature/top_p/top_k), seeded per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-done", default="")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="cap on image pixels (image_processor size longest_edge); 0 = model default")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    assert torch.cuda.device_count() == 1

    proc = AutoProcessor.from_pretrained(args.model_dir)
    if args.max_pixels:
        proc.image_processor.size["longest_edge"] = args.max_pixels
        proc.image_processor.max_pixels = args.max_pixels
    tok = proc.tokenizer
    tok.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    think_end = tok.convert_tokens_to_ids("</think>")
    assert isinstance(think_end, int) and think_end != tok.unk_token_id
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])

    items = pick(args.dataset, args.data_root)
    if args.limit:
        items = items[:args.limit]
    done = set()
    for f in filter(None, args.skip_done.split(",")):
        if os.path.exists(f):
            done |= {json.loads(l)["image_id"] for l in open(f)}
    items = [it for it in items[args.shard_index::args.num_shards] if it["image_id"] not in done]
    print(f"{args.model_tag} {args.dataset} shard {args.shard_index}/{args.num_shards}: {len(items)} items, "
          f"batch {args.batch_size}", flush=True)

    out_f = open(os.path.join(args.out_dir, "gen.jsonl"), "w")
    t0 = time.time()
    n_tok = 0
    def run(batch, imgs, b0):
        """Generate for a batch; on CUDA OOM split it in halves (down to single items).
        Returns [(prompt ids, generated ids after the padded prompt)] in batch order."""
        texts = [prompt_text(proc, im, it["question"]) for im, it in zip(imgs, batch)]
        inputs = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
        Pm = inputs["input_ids"].shape[1]
        try:
            if args.sample:
                torch.manual_seed(args.seed * 100_003 + args.shard_index * 10_007 + b0)
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True)
            else:
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     temperature=None, top_p=None, top_k=None)
        except torch.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            h = len(batch) // 2
            print(f"OOM at batch start {b0} (size {len(batch)}): splitting", flush=True)
            return run(batch[:h], imgs[:h], b0) + run(batch[h:], imgs[h:], b0 + h)
        res = [(inputs["input_ids"][j][inputs["attention_mask"][j].bool()], out[j, Pm:].tolist()) for j in range(len(batch))]
        del out, inputs
        torch.cuda.empty_cache()
        return res

    for b0 in range(0, len(items), args.batch_size):
        batch = items[b0:b0 + args.batch_size]
        imgs = [Image.open(os.path.join(args.data_root, it["file"])).convert("RGB") for it in batch]
        for j, (it, (p_ids, seq)) in enumerate(zip(batch, run(batch, imgs, b0))):
            single = build_prompt(proc, imgs[j], it["question"])["input_ids"][0]
            assert torch.equal(p_ids, single), "batched prompt differs from single-item prompt"
            cut = next((i for i, t in enumerate(seq) if t in eos), None)
            gen = seq if cut is None else seq[:cut + 1]
            n_tok += len(gen)
            te = gen.index(think_end) if think_end in gen else -1
            tail = gen[-1024:]
            seen, rep = set(), 0
            for i in range(len(tail) - 32):
                g32 = tuple(tail[i:i + 32])
                rep += g32 in seen
                seen.add(g32)
            rep32 = rep / max(1, len(tail) - 32)                          # share of repeated 32-grams in the last 1k tokens
            ans_text = tok.decode(gen[te + 1:], skip_special_tokens=True) if te >= 0 else ""
            pred, correct = score(it, ans_text, args.dataset) if te >= 0 else (None, False)
            rec = {"image_id": it["image_id"], "label": it["label"], "model_tag": args.model_tag, "dataset": args.dataset,
                   "prompt_len": int(len(p_ids)), "prompt_sha1": ids_hash(p_ids.cpu().numpy()),
                   "n_img_tokens": int((p_ids == model.config.image_token_id).sum()),
                   "gen_ids": gen, "n_gen": len(gen), "think_end": te, "hit_max": cut is None, "rep32_last1k": rep32,
                   "decoding": "sample" if args.sample else "greedy",
                   "pred": pred, "correct": bool(correct), "answer_text": ans_text[-300:],
                   "tail_text": tok.decode(gen[-60:], skip_special_tokens=False),
                   "question_type": it.get("question_type"), "category": it.get("category")}
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            el = time.time() - t0
            print(f"[{b0 + j}/{len(items)}] {el:.0f}s n_gen={len(gen)} think_end={te} hit_max={cut is None} "
                  f"pred={pred!r} label={it['label']!r} correct={correct} tok/s={n_tok / el:.1f}", flush=True)
    out_f.close()
    json.dump({"model_tag": args.model_tag, "dataset": args.dataset, "transformers": transformers.__version__,
               "n_items": len(items), "shard": [args.shard_index, args.num_shards], "max_new_tokens": args.max_new_tokens,
               "batch_size": args.batch_size, "decoding": "sample" if args.sample else "greedy", "seed": args.seed,
               "generation_config": model.generation_config.to_dict() if args.sample else None},
              open(os.path.join(args.out_dir, "run_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
