# Copyright (C) 2026 Xiaomi Corporation.
from typing import Any, Dict, List

import torch
from torch.utils.data.dataloader import default_collate
from transformers import AutoProcessor, Qwen3VLProcessor


class CustomCollateFlow2Act:
    """Apply the VLM chat template and collate the remaining tensors."""

    def __init__(self, processor_path: str | None = None) -> None:
        self.processor = (
            Qwen3VLProcessor.from_pretrained(processor_path, use_fast=False)
            if processor_path else AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        )
        self.processor.tokenizer.padding_side = "right"

    def _encode(self, messages):
        return dict(self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt", padding=True,
            images_kwargs={"do_resize": False}))

    def __call__(self, batch):
        lengths = {len(item["history_messages"]) for item in batch}
        if len(lengths) != 1:
            raise ValueError("All samples in a batch must use one memory_length")
        inputs = self._encode([item["messages"] for item in batch])
        count = lengths.pop()
        inputs["history_inputs"] = [
            self._encode([item["history_messages"][j] for item in batch]) for j in range(count)]
        payload = [{k:v for k,v in item.items() if k not in {"messages", "history_messages"}} for item in batch]
        inputs.update(default_collate(payload))
        for key in ("motion_2d", "motion_3d"):
            if key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].contiguous().view(torch.int32)
                elif inputs[key].dtype == torch.float16:
                    inputs[key] = inputs[key].contiguous().view(torch.uint16)
                else:
                    raise TypeError(f"{key} cache must use float32 or float16")
        return inputs
