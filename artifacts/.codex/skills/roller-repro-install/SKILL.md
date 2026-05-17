---
name: roller-repro-install
description: 安装和复用 Roller OSDI'22 artifact 的 TVM 0.8 CUDA 环境。用于 nnfusion/artifacts 仓库复现 README、构建 artifacts/.deps/tvm-0.8、配置 .venv、运行 roller/test_op_mp.py 或 GEMM 示例。
---

# Roller 复现安装

## 规则

- 用现有 `artifacts/.venv`。
- TVM 放 `artifacts/.deps/tvm-0.8`。
- 不改 `artifacts/roller` 源码。
- 生成物放 `/tmp` 或 `.deps/*/build`。

## TVM

在仓库根 `/root/nnfusion`：

```bash
mkdir -p artifacts/.deps
git submodule add https://github.com/apache/tvm artifacts/.deps/tvm-0.8
cd artifacts/.deps/tvm-0.8
git checkout 22ba6523cbd14fc44a1b093c482c1d02f3bc4fa5
git submodule update --init --recursive
```

若 GitHub 失败，可从已有 TVM 本地 clone；完成后必须修正：

```bash
git remote set-url origin https://github.com/apache/tvm
```

## 构建

```bash
cd /root/nnfusion/artifacts/.deps/tvm-0.8
mkdir -p build
cp cmake/config.cmake build/config.cmake
```

改 `build/config.cmake`：

```cmake
set(USE_CUDA /usr/local/cuda)
set(USE_LLVM OFF)
```

构建：

```bash
cd build
cmake ..
cmake --build . -j8
```

若工具损坏：

```bash
apt-get install --reinstall -y make cuda-nvrtc-dev-12-1 nsight-compute-2023.1.1 cuda-nsight-compute-12-1
```

## Python

```bash
cd /root/nnfusion/artifacts
source .venv/bin/activate
uv pip install 'numpy<2' decorator attrs tornado psutil scipy
export TVM_HOME=/root/nnfusion/artifacts/.deps/tvm-0.8
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH
```

验证：

```bash
python - <<'PY'
import tvm
print(tvm.__version__, tvm.runtime.enabled("cuda"))
PY
```

应输出 `0.8.dev0 1`。

## GEMM

```bash
cd /root/nnfusion/artifacts/roller
python test_op_mp.py --code_dir /tmp/roller_gemm4096_topk10/ \
  --smem_tiling --reg_tiling \
  --op matmul_expr --shape 4096 4096 4096 \
  --topk 10 --num_threads 1
```

单卡必须加 `--num_threads 1`。

## 备注

- RTX 3090/Ampere 上 `nvprof` 不支持，README 脚本的时间可能无效。
- 只需验证编译/执行时，可直接运行生成的可执行文件。
