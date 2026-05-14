---
name: install-roller-nnfusion
description: 安装、集成并验证 NNFusion artifacts 中 Roller/OSDI'22 artifact 仓库。用于用户要求安装该仓库、跑通 Roller、验证 gemm(4096,4096,4096)、测 topk=10 或对比 cuBLAS 性能时。
---

# 安装 Roller NNFusion

## 流程

1. 在仓库根目录确认文件：`README.md`、`setup.py`、`default.py`、`roller/`。
2. 确认 CUDA/GPU：`nvidia-smi`、`nvcc --version`。
3. 安装依赖：
   ```bash
   uv pip install --system --upgrade antares
   uv pip install --system 'numpy<2'
   uv pip install --system .
   ```
4. 集成 Roller 到 Antares：
   ```bash
   SITE=$(uv pip show antares | awk -F': ' '/^Location:/{print $2}')
   DST="$SITE/antares_core/backends/c-cuda/schedule/standard/default.py"
   [ -f "$DST.antares-original" ] || cp "$DST" "$DST.antares-original"
   cp default.py "$DST"
   ```
5. 若 C++ 编译报 `ERANGE/EINVAL` 等 errno 未定义，检查系统头：
   ```bash
   wc -c /usr/include/linux/errno.h /usr/include/asm-generic/errno.h
   apt-get install -y --reinstall linux-libc-dev
   ```

## 验证

跑通 `gemm(4096,4096,4096)`：

```bash
DEV_NAME=RTX3090 BACKEND=c-cuda STEP=10 COMPUTE_V1='- S = 4096; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares
```

说明：
- `DEV_NAME` 按机器改：如 `RTX3090`、`V100`。
- 日志出现 `found 10 results` 可视为 topk=10 候选生成。
- 报告性能用 ms：`sec * 1000`。

## 经验值

本机 RTX 3090 参考结果：
- Roller topk=10 final evaluator：约 `8.07 ms`，`17.02 TFLOPS`。
- cuBLAS SGEMM：约 `5.59 ms`，`24.58 TFLOPS`。

## 约束

- 不修改仓库源代码。
- Python 包安装使用 `uv pip`，需要系统环境时加 `--system`。
- 只允许安装包、集成 Antares 外部文件、修复系统依赖。
- 临时 benchmark 文件跑完删除。
