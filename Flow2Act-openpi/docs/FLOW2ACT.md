# Experiment contract and code map

## Independent experiments

`baseline`, `static`, `motion2d`, and `motion3d` each initialize independently
from the same local official π0.5 base. Each receives 80,000 optimizer updates.
Static adds DA3 alignment, motion2d adds future 2D supervision, and motion3d also
adds 3D supervision. The default `motion3d` therefore trains action + static +
motion2d + motion3d losses jointly, weighted 1 / 0.02 / 1.0 / 0.1.

Keep the original LIBERO delta/gripper labels, q01/q99 normalization, 32-dimensional
action capacity and padded-dimension supervision. There is no signed-gripper
training conversion, repeated-action expansion, or frequency-domain action loss.

## Tokens, memory, and supervision

The full-FOV current main/wrist prefix retains native prefix attention. Historical
image/state, current state, and 30 learned future queries are appended before the
first PaliGemma layer. Prefix tokens cannot read the new suffix; query attention
is causal. Action Expert sees current prefix and query KV, with direct
memory/state KV masked. Historical encodings are stop-gradient.

Memory length includes the current observation. The default 3 includes the current
frame and t−10,t−20. Longer memories continue the same cadence, with fixed-shape invalid placeholders
that do not consume valid positions. Online memory updates only after successful
decisions and is cleared at episode changes; retries do not insert duplicates.

The four shared static heads align student taps [8,11,14,17] to Teacher taps
[19,27,33,39] by FP32 cosine loss. Current unaugmented RGB views are passed jointly
to frozen DA3 as [B,V,…]. Teacher 18×18 features are resized to the student 16×16 grid.
Current training images retain shared photometric augmentation and complete FOV.

Motion readout uses the current main-camera grid and query hidden states:
[B,30,2,32,32], plus [B,30,3,32,32] for motion3d. Preserve all horizon, visibility,
in-bounds and valid-3D masks and the existing Smooth L1 balancing rules.

## Updates, checkpoints, and logs

GBS is the effective optimizer batch. Each device processes GBS/GA/GPUs samples
per microstep; clipping, AdamW, EMA, schedule, and global step advance once after
all GA microsteps. Learning rates are defined by actual updates:
update 1=5e-7, update 2000=5e-5, update 80000=5e-6.

Checkpoints at updates 5000, 10000,…,80000 retain live parameters, EMA, optimizer
and schedule counters, global step, RNG, sampler position, config, and norm stats.
The frozen Teacher is not checkpointed. Save commits atomically and requires
directory-rename support. All 16 checkpoints are retained by default.

Terminal progress reports percentage, elapsed time, update time, ETA and loss.
`metrics_flow2act.jsonl` records every optimizer update. ETA excludes the first
compiled update and includes later checkpoint time. Terminal output is not
automatically redirected to a file; capture it explicitly when required.

Fresh runs refuse nonempty output directories. Resume uses an explicitly selected
committed checkpoint and saved configuration; worker/logging/stop controls can be
overridden. `STOP_AFTER_UPDATES=20` runs a short gate without changing the 80k LR
timeline. The entry never changes batch or scientific settings after OOM.

## Modules

| File under `src/openpi/flow2act/` | Role |
| --- | --- |
| config_flow2act.py | Defaults, validation, model config, schedule |
| model_flow2act.py / gemma_flow2act.py | Derived π0.5 and attention/tap extensions |
| modules_flow2act.py | Shared alignment heads, motion readout and losses |
| teacher_flow2act.py | Pure JAX DA3-Giant, strict safetensors loading |
| teacher_contract_flow2act.py | Framework-free shapes shared by extraction and JAX |
| cache_flow2act.py | Shared cache schema and array names |
| data_flow2act.py | H30 index, raw videos/parquet, cache masks, deterministic sampler |
| training_flow2act.py | Initialization, accumulated updates, EMA, sharding, checkpoint save/restore |
| policy_flow2act.py | EMA restore, denoising, per-environment decision memory |
| client_flow2act.py | 30-step prediction and 10-step execution |
| artifacts_flow2act.py | Explicit CPU import of supplied local JAX EMA checkpoints |
| loss_dashboard_flow2act.html | Local interactive metric plots |

The native JAX model and registry stay unchanged. Files under `models_pytorch/`
are upstream import dependencies and compatibility sources; they are not a
second training chain.

## Copy-specific integration changes

Model mathematics, native base parameter paths, optimizer, sampling, and training
defaults are unchanged. The auxiliary parameter root and public artifact names
use `Flow2Act`/`flow2act` consistently; see the resource guide before reusing an
existing checkpoint or cache. Publication changes also include:
explicit resume selection; configurable simulator/interpreter/catalog paths;
per-run benchmark configs; resource-free test selection; documentation and
GitHub exclusions. No training/evaluation state was copied.
