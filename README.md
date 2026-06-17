# gpu-filler

将 GPU 利用率拉到指定的目标百分比，同时不影响真实任务。纯 CUDA + cuBLAS +
NVML 实现，不依赖 Python / PyTorch。

## 工作原理

- 每张选中的 GPU 启动一个 worker 线程。
- 每个 worker 在 **最低优先级 CUDA stream** 上循环跑 cuBLAS SGEMM，让 SM
  调度器把真任务的 kernel 排在前面。
- 每个工作 burst 之后 sleep 一段时间。一个简单的比例控制器每个 tick 读
  NVML 利用率，根据误差调整 work/idle 比例，使总利用率收敛到配置的目标。
- 真任务来的时候，NVML 总利用率会超过目标 → 控制器缩短 work_ms、增长
  idle_ms → 自动让出份额。

默认显存占用很小：每卡约 12 MiB（1024×1024 fp32 的 A/B/C 三块矩阵）。

## 编译

需要 CUDA Toolkit（nvcc、cuBLAS）和 NVML 头文件 / 库（装 NVIDIA 驱动就有）。

```bash
make                       # 生成 ./gpu-filler
make CUDA_PATH=/opt/cuda   # 如果 CUDA 不在默认路径
```

如果你的卡算力不在 Makefile 的 `GENCODE` 范围内，加一行对应的
`-gencode arch=...`。

## 用法

```bash
# 所有可见 GPU 拉到 70%
./gpu-filler

# 前 4 张 GPU 拉到 70%
./gpu-filler --gpus 4 --target 70

# 指定 GPU，统一目标
./gpu-filler --gpus 0,2,3 --target 60

# 每卡独立目标
./gpu-filler --spec 0:80,1:80,2:50,3:50

# 用更大的 SGEMM（每个 burst SM 占用更密、显存更多）
./gpu-filler --gpus 0 --target 70 --size 2048

# 看控制器每个 tick 的状态
./gpu-filler --gpus 0 --target 70 --verbose
```

用 `SIGINT`（Ctrl-C）或 `SIGTERM` 停止。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpus LIST` | 全部可见 GPU | `0,1,2` 这种列表，或一个数字 `4`（表示 0..3） |
| `--target PCT` | `70` | 1..99，每卡的默认目标利用率 |
| `--spec LIST` | — | `0:70,1:50,...`，覆盖列出 GPU 的 `--target` |
| `--size N` | `1024` | NxN SGEMM 矩阵大小；越大单次 burst 越密、显存越多 |
| `--tick MS` | `200` | 控制器 tick 间隔 |
| `--verbose` | 关 | 每卡每 tick 打一行日志 |

## "让位"机制怎么实现的

两层叠加：

1. **Stream 优先级**：`cudaStreamCreateWithPriority(..., lo_pri)` 把我们的
   GEMM kernel 放到 SM 调度队列底部。真任务用默认优先级 stream 提交时会被
   排在前面。
2. **闭环占空比控制**：NVML 读到的是总利用率（我们 + 真任务）。超目标就缩
   work_ms，低目标就涨 work_ms。稳态下：`空转份额 = 目标 - 真任务份额`。

如果你想要"硬上限"而不是软反馈，可以叠 CUDA MPS：

```bash
sudo nvidia-cuda-mps-control -d
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30 ./gpu-filler --gpus 4 --target 70
```

## 注意事项

- **NVML 利用率是时间占比，不是算力密度**。它表示采样窗口内有 kernel 运行
  的时间百分比。这正是大多数监控面板显示的指标，通常也正是你想拉高的，但
  别和 TFLOPS 利用率混淆。
- **显存不会动态归还**。共享卡上保持默认 `--size` 即可，避免把真任务挤
  OOM。
- **Stream 优先级在老架构上是建议性的**。Volta+ 才是稳定生效的。
- **多进程跑同一张卡**：每个实例自己跑反馈环。两个实例打同一张卡会互相
  抢、来回震荡——请用一个多卡实例。

## 作为服务运行

仓库里附了 `gpu-filler.service` 模板。`make` 之后：

```bash
sudo cp gpu-filler /usr/local/bin/
sudo cp gpu-filler.service /etc/systemd/system/
# 按你的部署改 unit 里的 ExecStart
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-filler
journalctl -u gpu-filler -f
```

## 项目结构

```
gpu-filler/
├── Makefile
├── README.md
├── gpu-filler.service
└── src/
    └── gpu_filler.cu
```
