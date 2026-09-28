#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trt=${TRTEXEC:-$(command -v trtexec || true)}
if [[ -z "$trt" && -x /usr/src/tensorrt/bin/trtexec ]]; then trt=/usr/src/tensorrt/bin/trtexec; fi
[[ -x "$trt" ]] || { echo 'Set TRTEXEC to TensorRT 10.7 trtexec.' >&2; exit 1; }
for height in 512 480; do
    model="spnet_${height}_640"
    [[ -f "$model.onnx" ]] || { echo "Missing $model.onnx; run export_onnx.sh first." >&2; exit 1; }
    [[ ! -e "$model.engine" ]] || { echo "Refusing to overwrite $model.engine; move it aside first." >&2; exit 1; }
    "$trt" --onnx="$model.onnx" --saveEngine="$model.engine" --fp16 \
        --shapes="rgb:1x3x${height}x640,depth:1x1x${height}x640,mask:1x1x${height}x640"
done
