# VLM-SinkQuant

[English](README.md) | 中文

**多模态大模型量化到 4 bit 时，视觉 token 需要特殊对待吗？**

大语言模型的量化方法通常会特殊保护 attention sink token，例如让第一个 token 的 KV cache 保持全精度。多模态大模型的图像里也有 sink，称为视觉 sink：少数几个图像 token 拿走了图像部分很大一部分注意力。本仓库研究的问题是：把多模态大模型的语言模型部分量化到 4 bit（逐 token INT4、MXFP4、NVFP4）时，视觉 sink 是否也需要同样的保护。仓库包含代码和单元测试，主要结论见下文。

- 模型：Qwen3-VL-8B-Thinking、Qwen3-VL-2B-Thinking（弱 sink 对照）、InternVL3.5-8B
- 评测：V\*、POPE、MathVista（按类别分层抽取的 150 题子集）
- 量化对象：语言模型全部线性层的输入激活，以及 KV cache（模拟量化，权重保持 BF16），分别测试加与不加 Hadamard 旋转

## 主要结论

| 问题 | 结果 |
|---|---|
| 已有工作报告的视觉 sink 统计能复现吗？ | 三个模型在 V\* 上都能复现。常用的「注意力超过均匀分配 100 倍」判据会随图像 token 数变化，所以只有在 token 数相同时，sink 数量才能直接比较。 |
| 量化格式如何改变激活量化误差？ | 逐 token INT4 的误差几乎全部来自同一行的小值被大缩放系数舍入成零。MXFP4 下，sink 行的误差主要来自大值本身的失真：Qwen3-VL-8B 为 90.8%，InternVL3.5-8B 为 79.7%，弱 sink 的 2B 模型为 28.2%。格式改变的是误差机制，不是误差的相对大小。 |
| 视觉 sink 在 KV cache 里特殊吗？ | 量化视觉 sink 的 K/V 对输出的扰动，只有量化同等注意力的普通 token 时的 6–20%。决定量化敏感度的是 token 获得的注意力份额，而不是它是不是 sink。 |
| 端到端地保护视觉 sink 有用吗？ | 在相同的 FP16 保留预算下，在 V\* 和 POPE 上保护视觉 sink 都不比保护随机图像 token 更好。在 KV cache 中保护唯一的文本 sink token 则在 POPE 上有效。 |
| 激活 INT4 量化会让模型「看不见」图像吗？ | 只对图像行做逐 token INT4 量化，会让 POPE 漏检率从 0.25 升到 0.77，行为接近输入纯灰图像时。Hadamard 旋转能完全消除这一影响。在 MXFP4、NVFP4 和旋转后的 INT3 下，剩余损伤主要集中在文本行。 |
| KV 量化误差会在长推理中累积吗？ | 不会。在 MathVista 上约 1250 token 的思考链中，每个位置受到的影响不随长度增长；而模型对图像的依赖从 0.64 降到 0.04。 |

## 目录结构

```
src/sinkquant/   quant.py      INT4 / MXFP4 / NVFP4 模拟量化器、Hadamard 与随机正交旋转、
                               P1/P2 误差分解
                 sinkstats.py  视觉 sink 检测与离群值统计
                 evalstats.py  配对 bootstrap 置信区间、答案选项上的 KL 散度
                 e2summary.py  保护实验的结果汇总
scripts/         e0_*  sink 存在性         e05_* 误差路径分解      e2_* 等预算保护
                 e3_*  图像行与文本行对比  e5_*  长思考链          smoke_* 模型加载检查
                 setup_data/  MathVista 下载（sha256 校验）与子集构建
tests/           量化器、统计与检测器的单元测试（25 个，仅需 CPU）
```

## 运行

实验环境：torch 2.11（CUDA 12.8）、transformers 4.57.1，GPU 为 RTX 3090 / RTX A6000。

```bash
pip install torch transformers numpy pillow pyarrow pytest
PYTHONPATH=src python -m pytest -q tests
```

每个脚本的参数都写在文件开头的说明里，例如：

```bash
PYTHONPATH=src python scripts/e0_sink_stats.py --model-dir M --model-tag T --coco-img DIR --coco-ann FILE \
    --n-images 200 --seed 0 --upscale 1 --out-dir OUT
PYTHONPATH=src python scripts/e2_protect.py --model-dir M --model-tag T --vstar-root V --e0-run-dir RUN/e0 \
    --formats int4_tok,mxfp4 --settings kv,act --out-dir OUT
```

仓库不含模型和数据集，请从官方页面下载并遵守各自的许可协议。仓库也不含逐题的原始结果文件。

## 许可

本仓库代码采用 MIT 许可。模型和数据集沿用各自的许可。
