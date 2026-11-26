# AGENTS.md — OpenPI JAX π0.5 / Flow2Act

## 范围

本仓库保留原生 JAX π0.5 和独立 JAX Flow2Act 两条训练链路。
本项目维护 Flow2Act 实验与其原生 OpenPI 依赖。

- 原生 JAX 模型、训练入口与全局 registry 保持不变；新增实验放在
  `src/openpi/flow2act/`，模块和入口使用 `_flow2act` 后缀，类名使用 `Flow2Act` 前缀。
- 不回退、覆盖或批量格式化用户的其他修改。
- 保留 JAX 训练、策略服务、LIBERO/LIBERO-Plus 评测、可视化及其依赖。
- 上游 `models_pytorch/` 的基础模块是原生 `models/model.py` 的直接导入依赖；
  不可按名字批量删除。Torch 也用于 LeRobot 数据加载和独立数值对照测试。
- 评测通过显式 benchmark 路径生成每次运行的配置。
- 不将数据、权重、缓存、虚拟环境、存储软链接或训练/评测结果加入版本控制。

## 模型与数据约定

- 不静默改变数据顺序、相机顺序、seed、基础模型、Teacher、checkpoint、loss、
  训练步数、batch、GA、LR、归一化、action 表示或推理协议。
- 原生 JAX 链路遵循其原生配置；以下默认值只适用于 Flow2Act。
- `baseline/static/motion2d/motion3d` 为独立实验，从本地官方 `pi05_base` 初始化。
- LIBERO4Mix 固定 `10 -> goal -> object -> spatial`；保留原生 delta/夹爪、
  32 维动作容量与 padding 监督、q01/q99 归一化、flow-matching MSE。
- Action horizon=30，Future queries=30，execute horizon=10，
  memory length=3（当前观测加 t−10、t−20）；历史间隔10步，episode 切换清空历史。
- 使用完整 H30 action 窗口，不 repeat-last。动作标签为 `t…t+29`，
  motion 标签为 `t+1…t+30`，遵循各自边界与有效性 mask。
- 保持完整 FOV 和 main/wrist 两个视角；冻结 DA3 当前双视角按 `[B,V,...]` 联合前向，
  Teacher 读取未增强 RGB。Teacher 不进入 optimizer、EMA 或 checkpoint。
- 当前 prefix 不读取新增 suffix；Future queries 从第0层参与编码，使用因果注意力。
  Action Expert 读取当前 prefix 和 query KV，屏蔽新增 memory/state 的直接 KV。
- 默认单机单 JAX 进程、4卡/FSDP4、有效 GBS256、GA1；
  每卡 microbatch=`GLOBAL_BATCH_SIZE / GA / NUM_GPUS`。
- 默认30,000次 optimizer 更新，每5,000次保存；warmup2000次从5e-7到5e-5，
  cosine 衰减至第30,000次的5e-6；EMA0.999、seed42。
- 不为避免 OOM 自动改变用户指定配置；如实报告验证范围及失败原因。

## 文件、权重与 checkpoint

- 数据与缓存通过仓库 `data/` 下的入口读取。正式训练离线运行，不下载或转换权重，
  不反向 import 外部研究项目源码。独立 Teacher 对照工具的显式参考模式除外。
- `preprocess/` 使用独立 Torch 环境和仓库内 Track4World 子集生成离线缓存；
  JAX 训练及推理不得导入预处理模型。缓存导出与读取共享格式约定。
- 已有 native-UV 缓存直接只读复用，按 metadata 识别旧文件后缀并严格校验语义；
  不因框架、memory、超参数或发布改名而重算标签，不更改原缓存文件或权限。
- 训练首次缺失 norm 时自动计算，已有时校验复用。
- 显式本地权重导入保留原 tensor、norm、stage 和 memory，只导出 EMA 推理包；
  不将旧 memory=1 权重标作 memory=3。
- 基础权重严格核对 key、shape、dtype、finite，不用随机值填补缺失参数，不计算 SHA。
- checkpoint 保存训练参数、EMA、optimizer/scheduler、global step、RNG、sampler 位置、
  完整配置和 norm stats；不保存冻结 Teacher，不增加 marker、receipt 或完整性 manifest。
- 默认输出目录由超参数生成，允许 `OUTPUT_DIR` 覆盖。fresh 运行拒绝非空目录，不生成随机 Run ID。
- 训练、部署、评测的具体协议见 `train_scripts/README_flow2act.md` 和实际保留入口。

## 安全与输出权限

- 正在执行的训练、评测及其依赖不得误删、重启或中断。
- 未明确授权时，不删除权重、数据、cache、训练日志或评测结果。
- 不递归删除/修改权限于仓库、数据集、checkpoint/cache 根或宽泛 results 根，
  不跟随存储 symlink 清理内容。
- 保留 `set -Eeuo pipefail`、`umask 000`、拓扑/batch 打印和原始退出码。
  训练及评测持久输出最终0777；只对本次精确 `OUTPUT_DIR` 执行权限收尾：
  `find -P "${OUTPUT_DIR}" -xdev \( -type d -o -type f \) -exec chmod 0777 -- {} +`。
  成功运行但权限收尾失败时整体失败；Python 原子发布 rename 后显式 chmod。
- 默认入口和文档使用仓库相对路径，不写机器专用挂载点或用户名。
- 不增加 source bundle、Run ID、marker、receipt、watchdog 或复杂 shell helper。

## 验证

- Shell 改动运行 `bash -n`；Python 完成 compile/import 和定向 pytest/ruff。
- 清理代码时确认两条 JAX 链路没有对已删模块的导入，原生 JAX 与 Flow2Act 核心逻辑保持不变。
- 模型/数据改动按真实样本、geometry/mask、B=1 前反向、多卡小步及20次更新逐级验证。
- 未执行的 GPU、多机、真实仿真或完整训练明确标为未验证。
