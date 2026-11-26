# Third-party attribution

The root license does not replace component or pretrained-model licenses.
No weights, datasets or simulator checkouts are distributed in this source copy.
The Track4World source/configuration subset is included for standalone preprocessing.

| Component | Included use and attribution |
| --- | --- |
| OpenPI | Native JAX model/training/data utilities and client SDK. Physical Intelligence; Apache-2.0 text retained in LICENSE. Source: https://github.com/Physical-Intelligence/openpi |
| Big Vision / SigLIP / Gemma code | Upstream copyright/license headers are retained in the corresponding OpenPI model files. Gemma terms are retained in LICENSE_GEMMA.txt. |
| DA3 / DINOv2 | JAX Teacher follows the DA3/DINOv2 feature architecture and numerical contract; attribute the original DA3 and Meta DINOv2 authors. The Teacher artifact remains separately licensed. Sources: https://github.com/ByteDance-Seed/Depth-Anything-3 and https://github.com/facebookresearch/dinov2 |
| Track4World / utils3d | Preprocessing runtime in third_party/Track4World/ and the adapted model in preprocess/track4world_model_flow2act.py. Full license and dependency notices are retained at third_party/Track4World/LICENSE.txt. Source: https://github.com/TencentARC/Track4World and https://github.com/jiah-cloud/utils3d. This does not relicense external model artifacts as Apache-2.0. |
| LIBERO / LIBERO-Plus | External datasets and simulator dependencies. Obtain the sources separately and retain their licenses. |
| Plotly.js | Optional local dashboard dependency, not bundled. Downloaded version 2.35.2 retains its own MIT license header and upstream notices. |

The JAX model, Teacher, and optimizer do not import another research repository.
The standalone cache exporter uses the included Torch preprocessing runtime.
The optional standalone reference-verification adapter is explicitly selected
and is not part of training or deployment.

Publication modifications to the development chain concern entrypoint/resource
portability, test selection, documentation, and exclusion rules. Original
copyright headers are retained. Consult the supplied source and model terms
before redistributing external weights or derived artifacts.
