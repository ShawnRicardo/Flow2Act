"""Optional device placement for the unchanged frozen FP32 DA3 teacher."""
from contextlib import nullcontext
import torch


@torch.no_grad()
def teacher_features_flow2act(teacher, rgb, *, device, cpu_offload=False,
                                    compute_device="cuda"):
    device=torch.device(device)
    if compute_device not in ("cpu","cuda"):
        raise ValueError("teacher_compute_device must be cpu or cuda")
    if getattr(teacher,"mmap_weights",False):
        if compute_device != "cpu" or cpu_offload:
            raise ValueError("Shared DA3 weights require CPU compute without GPU offload")
        if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in teacher.parameters()):
            raise ValueError("Shared DA3 weights must remain CPU FP32")
    if compute_device=="cpu":
        if cpu_offload:
            raise ValueError("CPU teacher compute cannot also enable GPU-between-batch offload")
        teacher.to(device="cpu")
        # The existing DA3 CPU path uses FP32 operators without CUDA autocast.
        # Only the detached targets move to the student's actual device.
        with torch.autocast("cpu",enabled=False):
            features=teacher(rgb.to(device="cpu",dtype=torch.float32))
        return tuple(value.detach().to(device=device) for value in features)
    if not cpu_offload:
        return teacher(rgb)
    teacher.to(device=device)
    # Disable only autocast's retained weight-copy cache. Each operator still
    # receives the same FP16 casts of the unchanged FP32 teacher parameters.
    context=(torch.autocast("cuda",dtype=torch.float16,cache_enabled=False)
             if device.type=="cuda" else nullcontext())
    try:
        with context:
            features=teacher(rgb)
    finally:
        teacher.to(device="cpu")
        if device.type=="cuda":
            torch.cuda.empty_cache()
    return features
