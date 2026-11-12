#!/bin/bash
# MathVista testmini (AI4Math/MathVista @2b6ad694, CC-BY-SA-4.0): one parquet of 141,568,126 bytes,
# checked against the sha256 of the pinned revision; then a stratified 150-question subset is written.
# Env: SINKQUANT_ROOT (default: repository root), PYTHON (needs pyarrow), HF_ENDPOINT (default huggingface.co).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
R=${SINKQUANT_ROOT:-$(cd "$HERE/../.." && pwd)}
PY=${PYTHON:-python}
EP=${HF_ENDPOINT:-https://huggingface.co}
REV=2b6ad69445fbb5695c9b165475e8decdbeb97747
F=data/testmini-00000-of-00001-725687bf7a18d64b.parquet
SHA=373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d
DST=$R/data/mathvista; mkdir -p "$DST/data"
curl -sSfL --retry 5 --retry-delay 5 -C - -o "$DST/$F" "$EP/datasets/AI4Math/MathVista/resolve/$REV/$F"
echo "$SHA  $DST/$F" | sha256sum -c -
"$PY" "$HERE/make_mathvista_subset.py" "$DST" 150 0
