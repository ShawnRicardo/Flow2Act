# Local JAX policy checkpoints

The README uses a local checkpoint until a download URL is supplied. No training
or evaluation entry downloads weights. A complete source directory contains:

```text
checkpoints/Flow2Act/source/checkpoint-30000/
  config_flow2act.json   # Or one archived config*.json with the full saved configuration
  norm_stats.json
  state/
    _CHECKPOINT_METADATA
    _METADATA
    ...                 # All Orbax/OCDBT data files and indexes
```

Keep the whole Orbax directory. A Torch `.ckpt`, raw official π0.5 base parameters,
or an individual OCDBT data file is not a trained Flow2Act policy checkpoint.
The tokenizer is prepared separately at `checkpoints/big_vision/paligemma_tokenizer.model`.

## Import an archived or supplied local policy

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/import_checkpoint_flow2act.py \
  --source checkpoints/Flow2Act/source/checkpoint-30000 \
  --output checkpoints/Flow2Act/checkpoint-30000
```

Import reads exactly EMA and the saved update on CPU. It checks the configured
parameter keys, shapes, dtypes and finite values, maps a uniquely identified
archived auxiliary root to `Flow2Act`, and retains the native OpenPI paths.
Tensor values and the norm-stat file are unchanged; no hashes are calculated.
The source directory is never modified. Output must be new, named after the
saved update, and is published atomically with 0777 permissions.

`--config /path/to/config.json` selects the configuration when more than one is
present. `--tokenizer-path` sets the deployment tokenizer path; its default is
the repository-relative path above. Unused resource pointers are changed to the
current repository defaults. Scientific configuration, including stage, horizons,
memory length, seed and auxiliary weights, is taken from the source checkpoint.
An old memory=1 checkpoint remains memory=1; import does not train memory modules.

The deployment directory contains `config_flow2act.json`, `norm_stats.json`, and
an Orbax `state/` with `ema` and `step` for inference. Optimizer and live weights
are not included in the deployment package.

## Policies trained in this repository

Use the committed run checkpoint directly:

```bash
.venv/bin/python scripts/serve_policy_flow2act.py \
  --checkpoint "outputs_flow2act/pi05/libero4mix/<run>/checkpoints/checkpoint-30000" \
  --port 8000
```

Only the checkpoint's EMA and normalization plus the tokenizer are needed for
inference. Teacher weights, motion caches, training videos and the original base
checkpoint are not loaded by the policy server. A client must provide episode IDs
and decision steps and follow the prediction-30/execution-10 protocol.
