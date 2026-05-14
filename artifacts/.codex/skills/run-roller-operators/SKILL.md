---
name: run-roller-operators
description: 在 NNFusion artifacts 的 Roller 仓库中运行不同 Antares/Roller 算子。用于用户要求跑 gemm、elementwise、reduce、pool、conv、depthwise conv、fused conv、softmax、脚本用例，或比较/报告算子性能时。
---

# 运行 Roller 算子

## 通用模板

```bash
DEV_NAME=RTX3090 BACKEND=c-cuda STEP=10 CHECK=1 COMPUTE_V1='- ...' antares
```

- `DEV_NAME` 按硬件改：`RTX3090`、`V100` 等。
- `CHECK=1` 做正确性检查；只测性能可去掉。
- `STEP=10` 做 10 步 tuning；只编译/快速验证可去掉或调小。
- 性能统一报告 `ms`：`sec * 1000`。

## 直接跑脚本

```bash
sh scripts/test_quick.sh
sh scripts/test_basic.sh
sh scripts/test_conv_fused_implicit_gemm.sh
sh scripts/test_depthwise_conv_fused.sh
```

长任务后台跑：

```bash
nohup sh scripts/test_basic.sh > scripts/log_basic 2>&1 &
```

## 常用单算子

GEMM 4096：

```bash
DEV_NAME=RTX3090 BACKEND=c-cuda STEP=10 COMPUTE_V1='- S = 4096; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares
```

Elementwise add：

```bash
CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})' antares
```

Reduce sum：

```bash
CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' antares
```

MaxPool：

```bash
CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] >=! input0[N, C, HO * 2 + KH, WO * 2 + KW] where HO in 6, WO in 6, KW in 2, KH in 2", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 12, 12]}})' antares
```

Direct Conv：

```bash
CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW] where HO in 30, WO in 30", { "input0": {"dtype": "float32", "shape": [16, 64, 32, 32]}, "input1": {"dtype": "float32", "shape": [256, 64, 3, 3]}})' antares
```

Depthwise Conv：

```bash
CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C, 0] where HO in 30, WO in 30", input_dict={"input0": {"dtype": "float32", "shape": [32, 16, 32, 32]}, "input1": {"dtype": "float32", "shape": [3, 3, 16, 1]}})' antares
```

## 查找更多用例

- 基础算子：`scripts/test_basic.sh`
- implicit GEMM fused conv：`scripts/test_conv_fused_implicit_gemm.sh`
- depthwise fused conv：`scripts/test_depthwise_conv_fused.sh`
- 不支持/边界用例：`scripts/not_supported.sh`

## 报告规则

- 记录命令、设备、`STEP`、best config、平均时间、GFLOPS/TFLOPS。
- 日志中 `Average time cost / run = X sec` 报为 `X*1000 ms`。
- 若出现编译或依赖错误，先确认安装技能 `install-roller-nnfusion` 的环境修复步骤。
