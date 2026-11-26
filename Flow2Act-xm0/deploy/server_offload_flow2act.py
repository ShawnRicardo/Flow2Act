"""Flow2Act XR0 server with optional BF16 weight offload for shared GPUs."""
from __future__ import annotations
import argparse
import json
import pickle
import socket
import struct
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from mmengine import Config
from transformers import Qwen3VLProcessor
from transformers.modeling_utils import no_init_weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xr0"))
sys.path.insert(0, str(ROOT))
from mibot.models.VLA.flow2act import Flow2ActXR0
from mibot.data.datasets.libero_dataset_flow2act import LiberoDatasetFlow2Act
from mibot.utils.io import resize_image, validate_stats, denormalize_action
from flow2act.memory import validate_execution_flow2act
from mibot.utils.checkpoint_contract_flow2act import (
    build_checkpoint_contract_flow2act, read_checkpoint_contract_flow2act,
    validate_checkpoint_contract_flow2act)


def place_policy_flow2act(model, device, *, cpu_offload_layers=0):
    """Keep selected VLM blocks on CPU between CUDA forwards, without casting.

    Selected blocks never enter the initial full-model CUDA allocation. Accelerate
    moves their original weights to CUDA for each forward and releases them again;
    attention activations, RoPE inputs and the autoregressive cache stay on CUDA.
    """
    count = int(cpu_offload_layers)
    if count != cpu_offload_layers or count < 0:
        raise ValueError("cpu_offload_layers must be a nonnegative integer")
    device = torch.device(device)
    if not count:
        return model.to(device)
    if device.type != "cuda":
        raise ValueError("CPU weight offload requires CUDA execution")
    layers = model.vlm.model.language_model.layers
    if count > len(layers):
        raise ValueError(f"Requested {count} offloaded layers; VLM has {len(layers)}")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Configure offload before moving the restored policy off CPU")
    from accelerate import cpu_offload

    selected = [(index, layers[index]) for index in range(count)]
    for index, _ in selected:
        layers[index] = torch.nn.Identity()
    try:
        model.to(device)
    finally:
        for index, layer in selected:
            layers[index] = layer
    for _, layer in selected:
        cpu_offload(layer, execution_device=device, offload_buffers=False)
    return model


def load_policy_flow2act(checkpoint, device="cpu", *, processor_path=None,
                                     vlm_config_path=None, allow_mechanical_smoke_checkpoint=False,
                                     cpu_offload_layers=0):
    """Strictly restore a Flow2Act policy and its complete architecture contract."""
    path = Path(checkpoint)
    if path.is_dir():
        candidates = (path / "checkpoint/mp_rank_00_model_states.pt", path / "mp_rank_00_model_states.pt")
        path = next((value for value in candidates if value.is_file()), candidates[0])
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    kinds = {"trained_policy"}
    if allow_mechanical_smoke_checkpoint:
        kinds.add("mechanical_training_smoke")
    contract = read_checkpoint_contract_flow2act(payload, allowed_kinds=kinds)
    params = dict(contract["model_config"])
    if params.pop("type") != "Flow2ActXR0":
        raise ValueError("Checkpoint is not a Flow2Act XR0 policy")
    config_path = vlm_config_path or (str(Path(processor_path) / "config.json") if processor_path else None)
    if config_path:
        params["vlm_config_path"] = config_path
    # Every persistent weight is strictly restored below. Skip random weight
    # initialization while still constructing real nonpersistent RoPE/mask buffers.
    with no_init_weights():
        model = Flow2ActXR0(**params)
    restored = build_checkpoint_contract_flow2act(
        model_config={"type": "Flow2ActXR0", **params}, method_config=model.geo_motion.config.to_dict(),
        vlm_config=model.vlm.config.to_dict(), experiment_config=contract["experiment_config"],
        action_stats=contract["action_stats"], kind=contract["kind"], batch_contract=contract.get("batch_contract"),
        optimizer_group_contract=contract.get("optimizer_group_contract"))
    validate_checkpoint_contract_flow2act(restored, contract)
    state = payload.get("module", payload.get("state_dict", {}))
    if not state:
        raise ValueError("Checkpoint contains no policy state_dict")
    state = {key.removeprefix("model."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    # assign=True restores storage without copying, but separate serialized names
    # can replace a tied Parameter with two objects. Restore the declared HF tie.
    model.vlm.tie_weights()
    model.eval().requires_grad_(False)
    place_policy_flow2act(model, device, cpu_offload_layers=cpu_offload_layers)
    model.checkpoint_global_step = int(payload.get("global_step", -1))
    return model, contract



def receive_exact_flow2act(conn, count):
    result = b""
    while len(result) < count:
        part = conn.recv(count - len(result))
        if not part:
            if not result:
                return None
            raise ConnectionError("Truncated request")
        result += part
    return result


class PolicyFlow2Act:
    def __init__(self, model_dir, checkpoint, device="cuda:0", *, processor_path=None,
                 stats_path=None,vlm_config_path=None,allow_mechanical_smoke_checkpoint=False,
                 cpu_offload_layers=0):
        self.model, contract = load_policy_flow2act(
            checkpoint, device, processor_path=processor_path, vlm_config_path=vlm_config_path,
            allow_mechanical_smoke_checkpoint=allow_mechanical_smoke_checkpoint,
            cpu_offload_layers=cpu_offload_layers)
        self.checkpoint_global_step = self.model.checkpoint_global_step
        self.checkpoint_kind = contract["kind"]
        self.device = torch.device(device)
        saved_processor = contract["experiment_config"]["data"]["params"]["processor_path"]
        self.processor = Qwen3VLProcessor.from_pretrained(processor_path or saved_processor, use_fast=False)
        self.processor.tokenizer.padding_side = "right"
        stats=contract["action_stats"]
        if stats_path and json.loads(Path(stats_path).read_text())!=stats:
            raise ValueError("External action normalization statistics differ from the saved checkpoint")
        if stats.get("gripper_encoding") != "signed_1_minus_2g_v1" or stats.get("action_encoding")!="libero_delta_xyz_axis_angle_gripper_signed_v1":
            raise ValueError("Signed new-method action statistics are required")
        mean, std = validate_stats(stats["mean"], stats["std"], self.model.action_shape[0])
        self.mean = torch.as_tensor(mean, device=self.device)
        self.std = torch.as_tensor(std, device=self.device)

    def _encode(self, observation):
        images = [observation["base"], observation["wrist_left"]]
        images = [Image.fromarray(np.asarray(image, dtype=np.uint8)) if not isinstance(image, Image.Image) else image for image in images]
        images = [resize_image(image.convert("RGB"), factor=32, max_pixels=90000) for image in images]
        messages = LiberoDatasetFlow2Act._messages(observation["language"], images)
        encoded = dict(self.processor.apply_chat_template([messages], tokenize=True, return_dict=True,
                       return_tensors="pt", padding=True, images_kwargs={"do_resize": False}))
        return {key:value.to(self.device) for key,value in encoded.items()}

    @torch.no_grad()
    def __call__(self, request):
        if request.get("request_type")=="metadata":
            return {"response_type":"policy_metadata","action_horizon":int(self.model.action_shape[0]),
                    "execute_horizon":int(self.model.execute_horizon),"memory_length":int(self.model.memory_length),
                    "motion_3d_enabled":bool(self.model.geo_motion.config.motion_3d_enabled),
                    "stage":self.model.stage,"future_horizon":self.model.future_horizon,
                    "architecture_version":"Flow2Act-XR0-checkpoint-v1",
                    "checkpoint_global_step":getattr(self,"checkpoint_global_step",-1),
                    "history_protocol":"real_decision_fifo_current_included","checkpoint_kind":self.checkpoint_kind}
        horizon = self.model.action_shape[0]
        validate_execution_flow2act(horizon, int(request["execute_horizon"]), self.model.execute_horizon)
        if int(request["memory_length"]) != self.model.memory_length:
            raise ValueError("Client memory_length differs from trained model")
        step = int(request["step_index"])
        if step < 0:
            raise ValueError("Decision step must be nonnegative")
        batch = self._encode(request)
        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (32,) or not np.isfinite(state).all():
            raise ValueError("Expected clean current LIBERO state[32]")
        batch["state"] = torch.as_tensor(state, device=self.device, dtype=torch.bfloat16)[None,None]
        batch["history_inputs"] = []
        states, valid, ages = [], [], []
        records = list(request.get("history", []))
        if len(records) > self.model.memory_length-1:
            raise ValueError("Too many decision-history records")
        # Training reads exactly t-M, t-2M, ...; only negative episode-start
        # offsets are unavailable. A missing positive offset is not padding.
        expected_steps = [step - self.model.execute_horizon * j
                          for j in range(1, self.model.memory_length)
                          if step - self.model.execute_horizon * j >= 0]
        if len(records) != len(expected_steps):
            raise ValueError("History must contain every available trained decision offset t-M*j")
        previous = step
        for record, expected_step in zip(records, expected_steps, strict=True):
            past_step = int(record["decision_step"])
            old = record["observation"]
            if not 0 <= past_step < previous:
                raise ValueError("History must contain unique past decisions, newest first")
            if past_step != expected_step:
                raise ValueError("History decision step differs from trained offset t-M*j")
            if int(old.get("step_index",-1)) != past_step:
                raise ValueError("History observation must carry its own decision step")
            if old.get("environment_id") != request["environment_id"] or old.get("episode_id") != request["episode_id"]:
                raise ValueError("History crossed an environment or episode")
            batch["history_inputs"].append(self._encode(old))
            old_state = np.asarray(old["state"],dtype=np.float32)
            if old_state.shape != (32,) or not np.isfinite(old_state).all():
                raise ValueError("Each past decision requires its own finite state[32]")
            states.append(old_state);valid.append(True);ages.append(step-past_step)
            previous = past_step
        while len(states) < self.model.memory_length-1:
            batch["history_inputs"].append(self._encode(request))
            states.append(np.zeros(32,dtype=np.float32));valid.append(False);ages.append(0)
        count = self.model.memory_length-1
        batch["history_state"] = torch.as_tensor(np.asarray(states, dtype=np.float32).reshape(1,count,1,32),
                                                 device=self.device, dtype=torch.bfloat16)
        batch["history_valid"] = torch.tensor([valid], device=self.device, dtype=torch.bool)
        batch["history_ages"] = torch.tensor([ages], device=self.device, dtype=torch.int64)
        batch["action"] = torch.zeros((1,horizon,32), device=self.device, dtype=torch.bfloat16)
        mask = torch.zeros_like(batch["action"], dtype=torch.int32);mask[..., :7] = 1
        batch["action_mask"] = mask
        # Replaying a failed RPC uses the same seed and raw decision payload.
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == "cuda" else []):
            torch.manual_seed(int(request.get("seed", 42)))
            pred = self.model.generate(batch)
        action = denormalize_action(pred.float(), self.mean, self.std) * mask
        if not torch.isfinite(action).all():
            raise FloatingPointError("Model returned nonfinite actions")
        return {"actions": action.cpu(), "episode_id":request["episode_id"], "decision_id":request["decision_id"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--processor-path")
    parser.add_argument("--stats-path","--norm-stats",dest="stats_path")
    parser.add_argument("--vlm-config-path")
    parser.add_argument("--allow-mechanical-smoke-checkpoint",action="store_true")
    parser.add_argument("--cpu-offload-layers", type=int, default=0,
                        help="Keep the first N VLM blocks on CPU between CUDA forwards; preserve checkpoint dtype")
    args = parser.parse_args()
    policy = PolicyFlow2Act(args.model_dir,args.checkpoint,args.device,
        processor_path=args.processor_path,stats_path=args.stats_path,vlm_config_path=args.vlm_config_path,
        allow_mechanical_smoke_checkpoint=args.allow_mechanical_smoke_checkpoint,
        cpu_offload_layers=args.cpu_offload_layers)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host,args.port));server.listen(1)
        print(f"POLICY_READY host={args.host} port={args.port} device={args.device} "
              f"checkpoint_global_step={policy.checkpoint_global_step} "
              f"memory_length={policy.model.memory_length} execute_horizon={policy.model.execute_horizon} "
              f"cpu_offload_layers={args.cpu_offload_layers}", flush=True)
        while True:
            conn,_ = server.accept()
            with conn:
                try:
                    while True:
                        head = receive_exact_flow2act(conn,4)
                        if head is None:break
                        body = receive_exact_flow2act(conn,struct.unpack(">I",head)[0])
                        result = policy(pickle.loads(body))
                        response = pickle.dumps(result,protocol=pickle.HIGHEST_PROTOCOL)
                        conn.sendall(struct.pack(">I",len(response))+response)
                except Exception as error:
                    print(f"Connection ended: {error}",flush=True)
                    traceback.print_exc()


if __name__ == "__main__":
    main()
