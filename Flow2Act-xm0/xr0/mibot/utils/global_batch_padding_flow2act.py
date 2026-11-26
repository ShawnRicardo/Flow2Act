"""Exact observation-batch means with neutral microbatch-one padded slots."""
from math import ceil


def batch_contract_flow2act(*, enabled, target, world_size, accumulation, micro_batch):
    if type(enabled) is not bool:
        raise ValueError("Exact effective batch enabled must be boolean")
    if not enabled:
        return {"enabled": False}
    if any(type(value) is not int for value in (target, world_size, accumulation, micro_batch)):
        raise ValueError("Effective batch fields must be integers")
    if target < 1 or world_size < 1 or micro_batch != 1 or accumulation != ceil(target / world_size):
        raise ValueError("Exact batch requires micro1 and GA=ceil(target/world_size)")
    return {"enabled": True, "target_global_batch": target, "world_size": world_size,
            "gradient_accumulation_steps": accumulation, "micro_batch_size": micro_batch,
            "native_padded_slots": world_size * accumulation,
            "weight_policy": "rotate_global_step_v1"}


def slot_weight_flow2act(*, target, world_size, accumulation, rank, micro_step, global_step):
    if any(type(value) is not int for value in (target, world_size, accumulation, rank, micro_step, global_step)):
        raise ValueError("Batch topology fields must be integers")
    batch_contract_flow2act(enabled=True, target=target, world_size=world_size,
                                  accumulation=accumulation, micro_batch=1)
    if not (0 <= rank < world_size and 0 <= micro_step < accumulation and global_step >= 0):
        raise ValueError("Rank/microstep/global step out of range")
    slots = world_size * accumulation
    slot = (micro_step * world_size + rank + global_step) % slots
    return slots / target if slot < target else 0.0
