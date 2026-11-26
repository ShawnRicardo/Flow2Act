"""DA3 weight shapes shared by extraction and JAX loading; no framework imports."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Flow2ActTeacherConfig:
    width: int = 1536
    heads: int = 24
    depth: int = 40
    hidden: int = 4096
    patch: int = 14
    source_grid: int = 37
    grid: int = 18
    alt_start: int = 13
    taps: tuple[int, ...] = (19, 27, 33, 39)
    rope_frequency: float = 100.0
    interpolate_offset: float = 0.1


DEFAULT_TEACHER_FLOW2ACT = Flow2ActTeacherConfig()


def expected_keys_flow2act(cfg):
    d, hidden, hd = cfg.width, cfg.hidden, cfg.width // cfg.heads
    shapes = {
        "camera_token": (1, 2, d),
        "cls_token": (1, 1, d),
        "pos_embed": (1, cfg.source_grid**2 + 1, d),
        "norm.weight": (d,),
        "norm.bias": (d,),
        "patch_embed.proj.weight": (d, 3, cfg.patch, cfg.patch),
        "patch_embed.proj.bias": (d,),
    }
    for i in range(cfg.depth):
        values = {
            "norm1.weight": (d,),
            "norm1.bias": (d,),
            "norm2.weight": (d,),
            "norm2.bias": (d,),
            "attn.qkv.weight": (3 * d, d),
            "attn.qkv.bias": (3 * d,),
            "attn.proj.weight": (d, d),
            "attn.proj.bias": (d,),
            "ls1.gamma": (d,),
            "ls2.gamma": (d,),
            "mlp.w12.weight": (2 * hidden, d),
            "mlp.w12.bias": (2 * hidden,),
            "mlp.w3.weight": (d, hidden),
            "mlp.w3.bias": (d,),
        }
        if i >= cfg.alt_start:
            values.update(
                {f"attn.{name}.{kind}": (hd,) for name in ("q_norm", "k_norm") for kind in ("weight", "bias")}
            )
        shapes.update({f"blocks.{i}.{k}": v for k, v in values.items()})
    return {"pretrained." + k: v for k, v in shapes.items()}
