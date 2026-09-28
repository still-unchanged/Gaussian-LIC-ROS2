# Model files (not stored in Git)

Obtain the SPNet `Large_300.pth` checkpoint from the original Gaussian-LIC/SPNet
release and place it in this directory. This repository does not include a
verified public checkpoint download URL. Do not substitute random weights.

With the environment from `env.sh` activated:

```bash
bash gaussian_lic/ckpt/export_onnx.sh
bash gaussian_lic/ckpt/build_trt.sh
```

Expected engines: `spnet_512_640.engine` and `spnet_480_640.engine`.
Build engines for the target GPU and TensorRT runtime, then run `colcon build`
to install them. Existing engines are never overwritten by the helper script.
`TRTEXEC` may select a specific TensorRT 10.7 executable.

LPIPS evaluation also requires `gaussian_lic/src/lpips/lpips_alex.pt`.
Recover the corresponding file from the original Gaussian-LIC repository or,
from `gaussian_lic/src/lpips`, run `python save_alex.py` in the Conda environment
(requires downloading torchvision pretrained weights). Rebuild to install it.

Weights, ONNX files and TensorRT engines remain local and are ignored by Git.
`env.sh` installs libraries; it does not download these models automatically.
