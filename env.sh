#!/usr/bin/env bash
# Ubuntu 22.04 x86_64 / ROS2 Humble / CUDA 12.6 / TensorRT 10.7.
# Execute to install; source to activate. Never enable nounset across ROS setup.
_gaussian_env_main() {
    local mode=${1:-install} repo conda_exe env_name env_prefix cuda_root
    repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return
    env_name=${GAUSSIAN_ENV_NAME:-gaussian_lic_ros2}
    cuda_root=${CUDAToolkit_ROOT:-/usr/local/cuda-12.6}
    case "$mode" in
        -h|--help)
            cat <<'HELP'
Usage:
  bash env.sh                 Create/update Conda env and install Python libraries.
  bash env.sh --system        Also install system/ROS/CUDA/TensorRT packages via apt.
  bash env.sh --check         Check dependencies only (no downloads or writes).
  bash env.sh --dry-run       Print install commands without running them.
  source ./env.sh             Activate environment and prepare native colcon build.

Defaults: GAUSSIAN_ENV_NAME=gaussian_lic_ros2, CUDA 12.6, TensorRT 10.7.
Requires an existing Conda installation. CONDA_EXE may specify its executable.
--system requires Ubuntu 22.04 x86_64 and configured ROS2/NVIDIA apt repositories.
It does not install/change GPU drivers. No C++ build or engine rebuild is run.
After installing: source ./env.sh; colcon build
HELP
            return 0 ;;
        install|--system|--check|--dry-run|--activate) ;;
        *) printf 'Unknown option: %s (see --help)\n' "$mode" >&2; return 2 ;;
    esac
    [[ $# -le 1 ]] || { echo 'Expected at most one option.' >&2; return 2; }
    [[ "$env_name" =~ ^[a-zA-Z0-9_-]+$ && "$env_name" != base ]] || {
        echo 'GAUSSIAN_ENV_NAME must be a non-base environment name (letters, digits, _ or -).' >&2; return 2;
    }
    conda_exe=${CONDA_EXE:-}
    if [[ ! -x "$conda_exe" ]]; then
        conda_exe=$(type -P conda || true)
    fi
    if [[ ! -x "$conda_exe" ]]; then
        for conda_exe in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
            [[ -x "$conda_exe" ]] && break
        done
    fi
    [[ -x "$conda_exe" ]] || { echo 'Conda not found. Install Conda or set CONDA_EXE to its executable.' >&2; return 1; }
    local conda_base
    conda_base=$("$conda_exe" info --base) || return
    env_prefix="$conda_base/envs/$env_name"

    if [[ "$mode" == --activate ]]; then
        [[ -x "$env_prefix/bin/python" && -f /opt/ros/humble/setup.bash ]] || {
            echo 'Environment or ROS2 Humble missing. Run bash env.sh (or --system) first.' >&2; return 1;
        }
        # Both Conda and ROS setup may reference unset variables.
        local restore_nounset=0 setup_status=0
        [[ $- == *u* ]] && restore_nounset=1
        set +u
        source "$conda_base/etc/profile.d/conda.sh" && conda activate "$env_prefix" &&
            source /opt/ros/humble/setup.bash || setup_status=$?
        if [[ $setup_status == 0 && -f "$repo/install/setup.bash" ]]; then
            source "$repo/install/setup.bash" || setup_status=$?
        fi
        if [[ $restore_nounset == 1 ]]; then set -u; fi
        [[ $setup_status == 0 ]] || return "$setup_status"
        export Torch_DIR="$env_prefix/lib/python3.10/site-packages/torch/share/cmake/Torch"
        export CUDAToolkit_ROOT="$cuda_root"
        export CUDACXX="$cuda_root/bin/nvcc"
        export MAKEFLAGS='-j1 -l1'
        export CMAKE_BUILD_PARALLEL_LEVEL=1
        # ROS/colcon use Ubuntu Python; `python` and explicit env Python remain Conda.
        export PATH="/usr/bin:/bin:$cuda_root/bin:$PATH"
        printf 'Ready: %s\nBuild: cd %q && colcon build\nPython tools: %q\n' "$env_name" "$repo" "$env_prefix/bin/python"
        return 0
    fi

    # Installation/check failures do not alter the caller shell's settings.
    (
        set -eo pipefail
        export PYTHONDONTWRITEBYTECODE=1
        run() {
            if [[ "$mode" == --dry-run ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
        }
        if [[ "$mode" == --system ]]; then
            . /etc/os-release
            [[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 && $(uname -m) == x86_64 ]] || {
                echo '--system supports Ubuntu 22.04 x86_64 only.' >&2; exit 1;
            }
            local trt_version='10.7.0.23-1+cuda12.6'
            local packages=(build-essential cmake pkg-config python3-dev python3-numpy python3-yaml
                python3-colcon-common-extensions python3-opencv libeigen3-dev libceres-dev
                libpcl-dev libopencv-dev libyaml-cpp-dev libboost-all-dev libffi-dev libcurl4-openssl-dev
                ros-humble-ros-base ros-humble-cv-bridge ros-humble-pcl-conversions
                ros-humble-tf2-eigen ros-humble-tf2-ros ros-humble-rosbag2 ros-humble-rosbag2-py
                ros-humble-rosbag2-storage-default-plugins ros-humble-visualization-msgs
                ros-humble-launch-ros cuda-toolkit-12-6
                "libnvinfer-dev=$trt_version" "libnvinfer-plugin-dev=$trt_version"
                "libnvonnxparsers-dev=$trt_version" "libnvinfer-bin=$trt_version")
            sudo apt-get update
            # apt reports unavailable repositories/versions before making changes.
            sudo apt-get install --dry-run --no-install-recommends "${packages[@]}"
            sudo apt-get install -y --no-install-recommends "${packages[@]}"
        fi
        if [[ "$mode" != --check ]]; then
            if [[ -e "$env_prefix" && ! -d "$env_prefix/conda-meta" ]]; then
                echo "Refusing to overwrite non-Conda directory: $env_prefix" >&2; exit 1
            fi
            if [[ -d "$env_prefix/conda-meta" ]]; then
                run "$conda_exe" install --yes --prefix "$env_prefix" 'python=3.10' pip
            else
                run "$conda_exe" create --yes --prefix "$env_prefix" 'python=3.10' pip
            fi
            # Keep NumPy <2 for compatibility with this project's OpenCV/ROS stack.
            run "$env_prefix/bin/python" -m pip install 'numpy==1.26.4'
            run "$env_prefix/bin/python" -m pip install 'torch==2.7.1' 'torchvision==0.22.1' \
                --index-url https://download.pytorch.org/whl/cu126
            run "$env_prefix/bin/python" -m pip install 'numpy==1.26.4' 'open3d==0.19.0' \
                'onnx==1.17.0' 'onnxruntime==1.20.1' 'opencv-python==4.10.0.84' 'PyYAML==6.0.2' 'rosbags==0.11.5'
            run "$env_prefix/bin/python" -m pip check
            if [[ "$mode" == --dry-run ]]; then
                printf 'Would check Python, ROS2, CUDA, TensorRT and GPU. Source env.sh to select Torch_DIR.\n'
                exit 0
            fi
        fi
        [[ -x "$env_prefix/bin/python" ]] || { echo "Missing environment: $env_prefix" >&2; exit 1; }
        "$env_prefix/bin/python" - <<'PY'
import sys
import torch, torchvision, numpy, open3d, onnx, onnxruntime, cv2, yaml, rosbags
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
assert torch.version.cuda == '12.6', torch.version.cuda
assert numpy.__version__ == '1.26.4', numpy.__version__
assert torch.cuda.is_available(), 'CUDA GPU unavailable: check NVIDIA driver/device access'
print('Python libraries OK; GPU:', torch.cuda.get_device_name(0))
PY
        [[ -f /opt/ros/humble/setup.bash && -x "$cuda_root/bin/nvcc" ]] || {
            echo 'Missing ROS2 Humble or CUDA 12.6 toolkit; configure apt repositories and run --system.' >&2; exit 1;
        }
        "$cuda_root/bin/nvcc" --version
        "$cuda_root/bin/nvcc" --version | /usr/bin/grep -q 'release 12.6,' || {
            echo 'Expected CUDA toolkit 12.6.' >&2; exit 1;
        }
        local missing=() dependency
        for dependency in build-essential cmake pkg-config python3-dev python3-numpy python3-yaml \
            libeigen3-dev libceres-dev libpcl-dev libopencv-dev libyaml-cpp-dev libboost-all-dev \
            libffi-dev libcurl4-openssl-dev ros-humble-pcl-conversions ros-humble-tf2-eigen \
            ros-humble-tf2-ros ros-humble-cv-bridge ros-humble-rosbag2 \
            ros-humble-rosbag2-storage-default-plugins ros-humble-visualization-msgs ros-humble-launch-ros; do
            [[ $(dpkg-query -W -f='${Status}' "$dependency" 2>/dev/null) == 'install ok installed' ]] || missing+=("$dependency")
        done
        if [[ ${#missing[@]} -gt 0 ]]; then
            printf 'Missing system package: %s\n' "${missing[@]}" >&2
            echo 'Use --system to install system dependencies.' >&2
            exit 1
        fi
        /usr/bin/python3 - <<'PY'
from pathlib import Path
import re
header = Path('/usr/include/x86_64-linux-gnu/NvInferVersion.h').read_text()
major, minor = [int(re.search(r'#define NV_TENSORRT_' + key + r'\s+(\d+)', header)[1])
                for key in ('MAJOR', 'MINOR')]
assert (major, minor) == (10, 7), f'Expected TensorRT 10.7, found {major}.{minor}'
for lib in ('libnvinfer.so', 'libnvinfer_plugin.so', 'libnvonnxparser.so'):
    assert (Path('/usr/lib/x86_64-linux-gnu')/lib).is_file(), lib
print('TensorRT development files OK')
PY
        source /opt/ros/humble/setup.bash
        /usr/bin/python3 -c 'import rclpy, rosbag2_py, cv_bridge, colcon_core; print("ROS2 Python dependencies OK")'
        printf 'Dependencies checked. Run: source %q\nThen: colcon build\n' "$repo/env.sh"
        echo 'Existing TensorRT engines are not regenerated; a different GPU/runtime may require rebuilding them.'
    )
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    _gaussian_env_main "${1:---activate}"
else
    _gaussian_env_main "$@"
fi
