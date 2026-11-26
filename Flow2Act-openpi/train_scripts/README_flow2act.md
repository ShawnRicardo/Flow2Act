# Flow2Act entries

Use [the repository README](../README.md) for installation and launch commands,
[the experiment contract](../docs/FLOW2ACT.md) for scientific defaults,
[resource formats](../docs/RESOURCES_flow2act.md) for local inputs, and
[the evaluation guide](../docs/EVALUATION_flow2act.md) for 8-GPU benchmarks.

- `train_pi05_libero_flow2act.sh`: one-host JAX training; hyperparameters and paths
  are exposed at the top and determine the output directory. Missing normalization
  statistics are computed automatically; existing statistics are validated and reused.

Preprocessing entries are in `preprocess/`; local policy import is
`scripts/import_checkpoint_flow2act.py`. Their resource layouts are described in
the linked guides.
