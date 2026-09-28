#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${CONDA_PREFIX:?Activate the Gaussian-LIC Conda environment first}"
[[ -f Large_300.pth ]] || { echo 'Missing ckpt/Large_300.pth; see ckpt/README.md' >&2; exit 1; }
"$CONDA_PREFIX/bin/python" export_onnx_512_640.py
"$CONDA_PREFIX/bin/python" export_onnx_480_640.py
