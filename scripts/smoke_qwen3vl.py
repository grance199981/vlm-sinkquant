"""Smoke test: load Qwen3-VL (Thinking) on one GPU, run one image+question.

Checks the things E0 will rely on: the model loads in BF16, eager attention returns
attention maps, hidden states are exposed, and an empty-thinking-block prompt
(2610.15855 App. A.1 style) yields a direct answer.

Usage: CUDA_VISIBLE_DEVICES=<GPU UUID> python smoke_qwen3vl.py <model_dir> <out_json>
"""
import json
import sys
import time

import torch
import transformers
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main(model_dir, out_path):
    assert torch.cuda.device_count() == 1, "expose exactly one GPU via CUDA_VISIBLE_DEVICES"
    gpu = torch.cuda.get_device_name(0)
    print("gpu:", gpu, flush=True)

    img = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(img).rectangle([150, 150, 300, 300], fill="red")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "What color is the square? Answer with one word."},
    ]}]

    proc = AutoProcessor.from_pretrained(model_dir)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Empty thinking block: the template may already open "<think>".
    text += "\n</think>\n\n" if text.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")

    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    t_load = time.time() - t0

    ids = inputs["input_ids"][0]
    is_img = ids == model.config.image_token_id
    with torch.no_grad():
        t0 = time.time()
        out = model(**inputs, output_attentions=True, output_hidden_states=True)
        t_fwd = time.time() - t0
        gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    answer = proc.batch_decode(gen[:, ids.shape[0]:], skip_special_tokens=True)[0]

    # Per-layer max |h| for visual vs text tokens: a first look at massive activations.
    hs = out.hidden_states
    max_vis = [h[0, is_img].abs().max().item() for h in hs]
    max_txt = [h[0, ~is_img].abs().max().item() for h in hs]

    rec = {
        "gpu": gpu,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "seq_len": int(ids.shape[0]),
        "n_visual_tokens": int(is_img.sum()),
        "n_layers_attn": len(out.attentions),
        "attn_shape_layer0": list(out.attentions[0].shape),
        "n_hidden_states": len(hs),
        "max_abs_visual_per_layer": [round(v, 2) for v in max_vis],
        "max_abs_text_per_layer": [round(v, 2) for v in max_txt],
        "answer": answer,
        "prompt_tail": text[-80:],
        "load_s": round(t_load, 1),
        "forward_s": round(t_fwd, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(json.dumps(rec, indent=1), flush=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
