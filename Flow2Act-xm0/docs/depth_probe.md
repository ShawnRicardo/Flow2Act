# 独立深度重建

这个实验在策略训练完成后读取一个 **Flow2Act policy checkpoint**。支持 `static`、`motion2d`、`motion3d`，以及这些分支中保存的 memory 长度；架构和四层 VLM taps 从 checkpoint 契约恢复。原生官方权重没有该契约，不能直接用作这个入口的输入。开发版本的策略和提取权重先按主 README 的导入命令转换到独立新路径。

完整 VLA、DA3 backbone 和原生 DualDPT 深度头均冻结，仅优化新 Adapter。它不修改策略 checkpoint，不作为动作推理的必需模块。

```text
冻结 VLM 的四层双视角特征 [B,2,8,8,2560]
  → 共享 LN → Linear(2560→512) → GELU
  → 双线性上采样至18×18
  → 两层共享3×3 Conv + GELU
  → 四层独立 Linear(512→3072)
  → 冻结 DA3 DualDPT
  → 主视角、腕部视角252×252深度
```

VLM 特征在 `no_grad` 下提取；Student 经过冻结深度头时保留输入梯度。Teacher 双视角联合编码生成伪深度和置信度。Adapter 独立保存为 Flow2Act 深度重建 checkpoint，与策略权重分开。

## 环境与输入

使用主 README 中的策略环境。在仓库根目录运行。必须显式设置 `POLICY_CHECKPOINT`，避免默认读取某个内部实验。可通过已激活 Conda 环境启动，或设置 `ENV_PATH` 指向环境目录。

必要资源：

- `DATA_ROOT`：LIBERO 四子集的 LeRobot 数据根目录，布局见主 README。
- `PROCESSOR_PATH`：本地 XR0 预训练目录，含 Qwen3-VL processor/tokenizer 资源。
- `TEACHER_WEIGHTS_DIR`：包含 `da3_anyview_giant_track4world_flow2act.safetensors` 和 `da3_dualdpt_track4world_flow2act.safetensors` 的目录。准备方法见数据准备文档。
- `POLICY_CHECKPOINT`：本仓库 Flow2Act 训练保存的 `.ckpt` 文件或 DeepSpeed checkpoint 目录，以及显式导入的策略文件。

数据使用 `probe_only` 模式：不读取动作标签、动作统计或 Motion cache；关闭图像增强。按 suite/task 分层、episode 级 seed=42 固定划分90%训练和10%验证，避免同一 episode 的帧跨 split。

## 四卡训练

```bash
conda activate flow2act
POLICY_CHECKPOINT=/path/to/trained-policy.ckpt \
DATA_ROOT=/path/to/libero_lerobot \
PROCESSOR_PATH=/path/to/Xiaomi-Robotics-0-Pretrain \
TEACHER_WEIGHTS_DIR=/path/to/da3_weights \
OUTPUT_ROOT=/path/to/results/probe/Flow2Act/DepthOnly \
NNODES=1 NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
GLOBAL_BATCH_SIZE=32 GRAD_ACCUM=1 \
MAX_STEPS=5000 WARMUP_STEPS=500 LR=1e-4 MIN_LR=1e-6 \
WEIGHT_DECAY=0.01 CLIP_GRAD_NORM=1.0 \
SI_LOG_WEIGHT=1.0 GRADIENT_WEIGHT=0.5 LOG_DEPTH_WEIGHT=0.1 \
VAL_FRACTION=0.1 VAL_EVERY_STEPS=500 VAL_MAX_BATCHES=0 SAVE_INTERVAL=500 \
NUM_WORKERS=4 SEED=42 \
bash train_scripts/train_flow2act_depth_probe.sh
```

八卡时设置 `NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。默认 GBS32/GA1，对应八卡每卡4，四卡每卡8。每卡 microbatch 为 `GBS/(NNODES×NPROC_PER_NODE×GA)`，必须整除。默认5000次更新、warmup500、AdamW betas=(0.9,0.95)、weight decay=0.01、梯度裁剪1.0。

下载和编译缓存使用各库默认位置，保留调用环境已有设置。

默认输出到 `results/probe/Flow2Act/DepthOnly/`，实验名包含源策略实验名、实际 step 和 probe 配方，可设置 `OUTPUT_ROOT` 或 `OUTPUT_DIR`。输出和模型缓存应放到容量充足的磁盘，或通过根目录 `results`、`checkpoints` 软链接分流。fresh run 拒绝覆盖已有实验。

多机读取 `MLP_WORKER_NUM/MLP_ROLE_INDEX/MLP_WORKER_0_HOST/MLP_WORKER_0_PORT`，或显式设置 `NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT`。每个节点运行同一入口。脚本不会安装环境或下载权重，默认使用已准备的本地资源。

## 训练目标

```text
L = SI_LOG_WEIGHT × SI-log
  + GRADIENT_WEIGHT × multiscale log-depth gradient
  + LOG_DEPTH_WEIGHT × unaligned log-depth L1
```

默认权重为1.0/0.5/0.1。SI-log 使用同一样本双相机共享的 log-scale offset；梯度 stride 为1/2/4；未中心化 log-depth 项约束尺度漂移。不计算 feature cosine 或 feature SmoothL1。

只使用 Teacher 有限正深度且 confidence>1 的像素，以 `(confidence−1)/confidence` 加权。相机、样本等权，空证据样本产生连接计算图的零损失。预测非有限或非正深度明确报错。Teacher 深度不是已知的米制真值。

## 日志、可视化与恢复

`log.log` 保存完整控制台输出；各节点另有 `console_nodeN_DepthOnly_flow2act.log` 和 torchrun 分 rank 原始日志。训练失败保留原始退出码。

每次验证保存四子集固定样本的双视角对比，位于：

```text
OUTPUT_DIR/val_depth/step_XXXXXXX/<suite>_episode..._frame.../
  comparison_DepthOnly_flow2act.png  # RGB / DA3彩色深度 / Student彩色深度 / 误差
  main_rgb.png, wrist_rgb.png
  raw_depth_DepthOnly_flow2act.npz  # 原始浮点深度、置信度、mask和RGB
  teacher_*_raw_depth.png, student_*_raw_depth.png
  depth_units_DepthOnly_flow2act.json
```

Teacher 和 Student 使用相同的 Teacher 色标。原始预测不进行尺度对齐。验证指标另记录原尺度、每样本双相机共享尺度对齐后的 AbsRel/RMSE/δ1，以及有效像素比例和各相机分项。默认 `VAL_MAX_BATCHES=0` 验证全部留出帧；仅在调试时限制该值。

Adapter checkpoint 为 `probe_step_XXXXXXX_DepthOnly_flow2act.pt`，包含 Adapter、optimizer、scheduler、更新计数、源策略契约、损失配置和 episode 划分。恢复时设置相同 `POLICY_CHECKPOINT`、配方与明确的 `RESUME_PROBE`；`MAX_STEPS` 表示最终总更新数，不是额外步数。可以使用同一输出目录，禁止把原混合损失 probe 当作 DepthOnly 恢复。不承诺数据迭代器逐位续接。

以上指标衡量相对 DA3 伪标签的一致性，不能单独证明真实米制深度精度、完整 DA3 特征恢复或策略成功率提升。

## CPU 验证

```bash
PYTHONPATH=.:xr0 python -m unittest \
  flow2act.test_depth_probe \
  xr0.tests.test_depthonly_entry_flow2act
```

这些测试覆盖纯深度梯度、冻结边界、mask、尺度、episode split、恢复契约、分布式参数和日志退出码，不等同于全量GPU训练验证。
