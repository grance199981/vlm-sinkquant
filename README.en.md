# VLM-SinkQuant

English | [中文](README.md)

**Do visual tokens need special treatment when multimodal LLMs are quantized to 4 bits?**

LLM quantizers routinely protect attention-sink tokens, for example by keeping the KV cache of the first token in full precision. Multimodal LLMs also develop *visual* sinks: a few image tokens that absorb a large share of the attention paid to the image. This repository asks whether those visual sinks deserve the same protection when the language model of an MLLM is quantized to 4 bits (per-token INT4, MXFP4, NVFP4). This repository contains the code and tests; the findings are summarized below.

- Models: Qwen3-VL-8B-Thinking, Qwen3-VL-2B-Thinking (weak-sink control), InternVL3.5-8B
- Benchmarks: V\*, POPE, MathVista (150-question stratified subset)
- What is quantized: activations of all language-model linear layers and/or the KV cache (fake quantization, weights in BF16), with and without Hadamard rotation

## Findings

| Question | Result |
|---|---|
| Do the visual-sink statistics of prior work reproduce? | Yes on V\* for all three models. The common "100× uniform attention" criterion depends on the number of image tokens, so sink counts are only comparable at matched token counts. |
| How does the format change activation-quantization error? | Per-token INT4 error is almost entirely small values zeroed by a large row scale. Under MXFP4, error on sink rows is mostly distortion of the large value itself: 90.8% (Qwen3-VL-8B) and 79.7% (InternVL3.5-8B) vs. 28.2% in the weak-sink 2B model. The format changes the mechanism, not the relative magnitude. |
| Are visual sinks special in the KV cache? | Quantizing their K/V perturbs the output only 6–20% as much as quantizing non-sink tokens that receive the same attention. Attention mass, not sink status, explains sensitivity. |
| Does protecting visual sinks help end to end? | With an equal FP16 budget, protecting visual sinks is never better than protecting random image tokens on V\* or POPE. Protecting the single text-sink token in the KV cache does help on POPE. |
| Does activation INT4 make the model "blind"? | Quantizing only image rows with per-token INT4 moves the POPE miss rate from 0.25 to 0.77, towards the behaviour on a gray image. A Hadamard rotation removes the effect. Under MXFP4, NVFP4 and rotated INT3 the residual damage concentrates on text rows. |
| Does KV-quantization error accumulate over long reasoning? | No. On MathVista thinking chains of about 1,250 tokens the per-position effect does not grow, while the model's reliance on the image decays from 0.64 to 0.04. |

## Layout

```
src/sinkquant/   quant.py      INT4 / MXFP4 / NVFP4 fake quantizers, Hadamard and random rotations,
                               P1/P2 error decomposition
                 sinkstats.py  visual-sink detectors and outlier statistics
                 evalstats.py  paired bootstrap intervals, KL over answer options
                 e2summary.py  summaries for the protection experiment
scripts/         e0_*  sink existence      e05_* error-path decomposition   e2_* equal-budget protection
                 e3_*  image vs. text rows e5_*  long thinking chains       smoke_* model loading checks
                 setup_data/  MathVista download (sha256-pinned) and subset construction
tests/           unit tests for quantizers, statistics and detectors (25 tests, CPU only)
```

## Running

Experiments ran with torch 2.11 (CUDA 12.8), transformers 4.57.1, on RTX 3090 / RTX A6000.

```bash
pip install torch transformers numpy pillow pyarrow pytest
PYTHONPATH=src python -m pytest -q tests
```

Each script documents its arguments in its docstring, for example:

```bash
PYTHONPATH=src python scripts/e0_sink_stats.py --model-dir M --model-tag T --coco-img DIR --coco-ann FILE \
    --n-images 200 --seed 0 --upscale 1 --out-dir OUT
PYTHONPATH=src python scripts/e2_protect.py --model-dir M --model-tag T --vstar-root V --e0-run-dir RUN/e0 \
    --formats int4_tok,mxfp4 --settings kv,act --out-dir OUT
```

Models and datasets are not included; download them from their official pages and follow their licenses. Raw per-item result files are not included.

## License

MIT for the code in this repository. Models and datasets keep their own licenses.
