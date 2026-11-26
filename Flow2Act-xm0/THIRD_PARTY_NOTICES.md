# Third-party code and weights

The root license does not replace the licenses of third-party source code or
model weights. No pretrained weights or datasets are distributed in this tree.

| Component | Location / use | Attribution and terms |
| --- | --- | --- |
| Xiaomi-Robotics-0 | `xr0/`, base policy and training infrastructure | Xiaomi Robotics; the upstream Apache-2.0 license is retained at the repository root. Source: https://github.com/XiaomiRobotics/Xiaomi-Robotics-0 |
| Track4World | `third_party/Track4World/`; native-UV model adaptation in `flow2act/model.py` | Tencent, 2026. The upstream license permits academic use and excludes commercial / production use; retain [the complete license and dependency notices](third_party/Track4World/LICENSE.txt). Source: https://github.com/TencentARC/Track4World |
| Depth Anything 3 | `flow2act/da3/` and `third_party/Track4World/track4world/nets/external/depth_anything_3/` | ByteDance, 2025; source files retain Apache-2.0 headers. Sources and model weights have their own terms: https://github.com/ByteDance-Seed/Depth-Anything-3 and https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1 |
| Pi3 / Pi3X, DINOv2, and the other geometry / tracking dependencies included by Track4World | `third_party/Track4World/track4world/nets/external/`, `nets/dinov2/`, and shared blocks | Original headers remain in the copied source. Consult the dependency terms bundled in [Track4World's license](third_party/Track4World/LICENSE.txt), including its Pi3 non-commercial terms. Flow2Act's exporter selects the DA3 route. |
| utils3d | `third_party/Track4World/utils3d/` | Source: https://github.com/jiah-cloud/utils3d ; copied as the runtime dependency accompanying the Track4World checkout. Retain source attribution and consult that project's terms for redistribution or use. |
| LIBERO / LIBERO-Plus | External datasets and simulator dependencies; evaluation code is adapted to them | Not bundled. Obtain them from their respective upstream repositories and follow their licenses. |

The frozen DA3 backbone and DualDPT files extracted by
`preprocess/extract_da3_weights_flow2act.py` come from the Track4World
`track4world_da3.pth` checkpoint. Extraction changes the container and removes key
prefixes; it does not grant a new model license or convert those weights into
independently licensed Flow2Act weights. The fixed depth decoder reconstructs
DA3 pseudo-depth, not ground-truth metric depth.

Flow2Act names its own modules, training entries and artifact formats separately
from these upstream projects. Track4World and DA3 names are retained for genuine
third-party attribution. Offline cache and weight-extraction helpers are included
in this project and do not require OpenPI.
