# Track4World runtime used for Flow2Act preprocessing

This is the source subset needed by the native-UV Motion cache exporter. The
upstream project is [TencentARC/Track4World](https://github.com/TencentARC/Track4World).
Its full license and dependency notices are retained in [LICENSE.txt](LICENSE.txt).
`utils3d` is the implementation used by that runtime, from
[jiah-cloud/utils3d](https://github.com/jiah-cloud/utils3d).

Only Python sources, runtime YAML/JSON configuration and a small runtime colormap
asset are included. Demos, old multi-camera teacher models, benchmark runners,
Grounded-SAM, checkpoints, datasets, results and Git metadata are excluded.
The preprocessing-specific model and decoder live in
`../../preprocess/track4world_model_flow2act.py` and
`../../preprocess/track4world_export_flow2act.py`.

The native-UV model retains the S16 decoder, stride-8 overlap, four iterations,
source self-slot and native low-resolution warm-start states. It exposes the
unmodified DA3 depth, confidence, intrinsics and poses used to lift the 2D tracks
into the anchor camera coordinates. No raw Track4World XYZ flow is used as the
3D training target.

Local adaptations retained here resolve DA3 Nested and ConvNeXt weights from a
local directory and use the PyTorch RoPE fallback when the optional extension is
not available. `track4world/paths.py` additionally accepts
`TRACK4WORLD_CHECKPOINTS_ROOT`; it defaults to `./checkpoints` in this directory.
No source or weight download occurs during a training rank's initialization.

See [resource preparation](../../docs/RESOURCES_flow2act.md) for the dedicated
preprocessing environment, public weight sources, extraction and cache export.
