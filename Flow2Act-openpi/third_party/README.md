# Runtime dependencies

`Track4World/` contains the source/configuration subset used by the standalone
motion-cache exporter, including its dependency notices and `utils3d` runtime.
JAX training and policy serving never import this preprocessing implementation.

LIBERO and LIBERO-Plus are installed separately under `libero/` and `LIBERO-plus/`.
Their datasets, assets and all model weights are external resources excluded by Git.
See [resources](../docs/RESOURCES_flow2act.md),
[evaluation setup](../docs/EVALUATION_flow2act.md), and
[third-party notices](../THIRD_PARTY_NOTICES.md).
