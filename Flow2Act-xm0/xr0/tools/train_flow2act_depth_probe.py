"""Depth-only reconstruction from a completely frozen Flow2Act XR0 policy."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "xr0")]
from flow2act.depth_probe import (
    Flow2ActDepthProbeConfig, Flow2ActDepthProbe,
    depth_only_losses_flow2act, encode_probe_teacher_depthonly_flow2act,
    depth_metric_sums_flow2act, finalize_depth_metrics_flow2act,
    stratified_episode_split_flow2act, episode_frame_indices_flow2act,
)

KIND = "independent_depth_adapter_depthonly_flow2act"
LOSS_KEYS = ("loss", "si_log", "multiscale_gradient", "log_depth")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-checkpoint", required=True)
    p.add_argument("--data-root", default=str(ROOT / "data/libero_lerobot"))
    p.add_argument("--processor-path", default=str(ROOT / "checkpoints/Xiaomi-Robotics-0-Pretrain"))
    p.add_argument("--teacher-weights-dir", default=str(ROOT / "checkpoints/da3_flow2act"))
    p.add_argument("--output-root", default=str(ROOT / "results/probe/Flow2Act/DepthOnly"))
    p.add_argument("--output-dir", default="")
    p.add_argument("--prepare-output", action="store_true", help="CPU-only launch preflight; print the resolved output directory")
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--si-log-weight", type=float, default=1.0)
    p.add_argument("--gradient-weight", type=float, default=0.5)
    p.add_argument("--log-depth-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-max-batches", type=int, default=0, help="0 validates all held-out frames")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--verify-frozen-policy", action="store_true", help="Smoke check: compare fixed-seed actions before and after adapter training")
    return p.parse_args()


def policy_metadata(path):
    from mibot.utils.checkpoint_contract_flow2act import read_checkpoint_contract_flow2act
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = (path / "checkpoint/mp_rank_00_model_states.pt", path / "mp_rank_00_model_states.pt")
        path = next((value for value in candidates if value.is_file()), candidates[0])
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    contract = read_checkpoint_contract_flow2act(payload, allowed_kinds={"trained_policy"})
    step = int(payload.get("global_step", -1))
    if step <= 0:
        raise ValueError("Depth reconstruction requires a trained Flow2Act policy with a positive global_step")
    return contract, step


def validate_arguments(args, world):
    if world <= 0 or args.global_batch_size <= 0 or args.grad_accum <= 0 or args.global_batch_size % (world * args.grad_accum):
        raise ValueError("GBS must be positive and divisible by total GPUs * GA")
    if args.max_steps < 1 or args.warmup_steps < 0 or not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.min_lr <= args.lr:
        raise ValueError("Invalid depth adapter step/lr settings")
    weights = (args.si_log_weight, args.gradient_weight, args.log_depth_weight)
    if any(not math.isfinite(x) or x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError("Depth loss weights must be finite/nonnegative with a positive total")
    if (args.seed < 0 or args.num_workers < 0 or args.val_max_batches < 0 or args.save_every < 0
            or args.val_every < 0 or args.log_every <= 0 or not math.isfinite(args.weight_decay)
            or args.weight_decay < 0 or not math.isfinite(args.clip_grad_norm) or args.clip_grad_norm <= 0
            or not 0 < args.val_fraction < 1):
        raise ValueError("Invalid depth adapter counts, split, weight decay or gradient clipping")


def resolved_output(args, contract, step, world):
    policy = Path(args.policy_checkpoint).expanduser().resolve()
    args.policy_checkpoint = str(policy)
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    method = contract["method_config"]
    # The source run identifies the policy experiment; the step comes from the
    # checkpoint itself, never from a filename or an overwritten sidecar config.
    run = policy.parent.name
    key = (f"{run}-step{step:07d}-{method['stage']}-L{method['memory_length']}-"
           f"{world}gpu-gbs{args.global_batch_size}-ga{args.grad_accum}-lr{args.lr:g}-"
           f"min{args.min_lr:g}-wu{args.warmup_steps}-steps{args.max_steps}-"
           f"si{args.si_log_weight:g}-gr{args.gradient_weight:g}-log{args.log_depth_weight:g}-"
           f"wd{args.weight_decay:g}-clip{args.clip_grad_norm:g}-s{args.seed}-vf{args.val_fraction:g}")
    if len(key.encode()) > 255:
        raise ValueError("Source run name is too long for the automatic output leaf; set OUTPUT_DIR explicitly")
    return Path(args.output_root).expanduser().resolve() / key


def check_output(output, *, prepare=False, node_rank=0):
    if not output.exists():
        return
    for entry in output.iterdir():
        console = re.fullmatch(r"console_node(\d+)_DepthOnly_flow2act\.log", entry.name)
        torchrun = re.fullmatch(r"torchrun_node(\d+)_DepthOnly_flow2act", entry.name)
        if prepare:
            # Other nodes can arrive first. Node 0 still refuses any trace of an
            # earlier node-0 launch; its console FD also prevents concurrent reuse.
            if node_rank != 0 and (console or torchrun or entry.name == "log.log"):
                continue
            if node_rank == 0 and ((console and int(console[1]) != 0) or (torchrun and int(torchrun[1]) != 0)):
                continue
        elif entry.name == "log.log" or console or torchrun:
            continue
        raise FileExistsError(f"Choose a fresh OUTPUT_DIR: {output}")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def visual_samples(dataset, validation_episodes):
    result, seen = [], set()
    for index in validation_episodes:
        episode = dataset.episodes[index]
        if episode["suite"] in seen:
            continue
        start = int(dataset.sample_ends[index - 1]) if index else 0
        stop = int(dataset.sample_ends[index])
        result.append(start + (stop - start) // 2)
        seen.add(episode["suite"])
    return result


def save_depths(output, step, student, teacher, confidence, rgb, *, suite, episode, frame):
    from PIL import Image, ImageDraw
    directory = output / "val_depth" / f"step_{step:07d}" / f"{suite}_episode{episode:06d}_frame{frame:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    student, teacher, confidence = (value[0].detach().float().cpu().numpy() for value in (student, teacher, confidence))
    rgb = rgb[0].detach().cpu().numpy().transpose(0, 2, 3, 1)
    valid = np.isfinite(teacher) & (teacher > 1e-6) & np.isfinite(confidence) & (confidence > 1)
    values = teacher[valid]
    low, high = np.percentile(values, (1, 99)).tolist() if values.size else [0.0, 1.0]
    high = max(high, low + 1e-6)
    # Fixed Teacher-derived ranges make this sample comparable across updates.
    anchors = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=np.float32)
    palette = np.stack([np.interp(np.linspace(0, 4, 256), np.arange(5), anchors[:, c]) for c in range(3)], -1).astype(np.uint8)

    def color(value, lower, upper):
        value = np.nan_to_num(value, nan=lower, posinf=upper, neginf=lower)
        indices = np.rint(np.clip((value - lower) / (upper - lower), 0, 1) * 255).astype(np.uint8)
        return palette[indices]

    np.savez_compressed(directory / "raw_depth_DepthOnly_flow2act.npz",
                        student=student, teacher=teacher, teacher_confidence=confidence, valid_mask=valid, rgb=rgb)
    height, width = teacher.shape[-2:]
    canvas = Image.new("RGB", (width * 4, (height + 22) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    maximum = max(float(np.max(student)), float(np.nanmax(np.where(valid, teacher, 0))), 1e-6)
    unit = maximum / 65535
    for view, camera in enumerate(("main", "wrist")):
        Image.fromarray(rgb[view]).save(directory / f"{camera}_rgb.png")
        panels = [np.asarray(Image.fromarray(rgb[view]).resize((width, height))),
                  color(teacher[view], low, high), color(student[view], low, high),
                  color(np.abs(student[view] - teacher[view]), 0, max((high - low) * 0.25, 1e-6))]
        for index, panel in enumerate(panels):
            if index in (1, 3):
                panel = np.where(valid[view, :, :, None], panel, 0)
            canvas.paste(Image.fromarray(panel), (index * width, view * (height + 22) + 22))
            draw.text((index * width + 3, view * (height + 22) + 3),
                      f"{camera}: {('RGB', 'DA3 teacher', 'Student adapter', 'Absolute error')[index]}", fill="black")
        for name, arrays in (("student", student), ("teacher", teacher)):
            encoded = np.rint(np.clip(np.nan_to_num(arrays[view]), 0, maximum) / unit).astype(np.uint16)
            Image.fromarray(encoded).save(directory / f"{name}_{camera}_raw_depth.png")
    canvas.save(directory / "comparison_DepthOnly_flow2act.png")
    write_json(directory / "depth_units_DepthOnly_flow2act.json",
               dict(png_depth_unit=unit, color_depth_range=[low, high],
                    color_range_source="Teacher valid depth percentiles 1/99, shared across cameras",
                    absolute_error_color_max=max((high - low) * 0.25, 1e-6),
                    scale_fitting=False, reference="DA3 pseudo-depth, not metric ground truth",
                    suite=suite, episode_index=episode, frame_index=frame))


def main():
    args = arguments()
    launch_world = int(os.environ.get("NNODES", "1")) * int(os.environ.get("NPROC_PER_NODE", "8"))
    world = launch_world if args.prepare_output else int(os.environ.get("WORLD_SIZE", str(launch_world)))
    validate_arguments(args, world)
    source_contract, source_step = policy_metadata(args.policy_checkpoint)
    output = resolved_output(args, source_contract, source_step, world)
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if args.prepare_output or rank == 0:
        check_output(output, prepare=args.prepare_output, node_rank=node_rank)
    if args.prepare_output:
        output.mkdir(parents=True, exist_ok=True)
        print(output)
        return
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Depth adapter training requires CUDA; use --prepare-output for CPU launch checks")
    if world > 1 and "RANK" not in os.environ:
        raise ValueError("Use the Shell entry or torchrun for multi-GPU training")
    visible = torch.cuda.device_count()
    if visible != int(os.environ.get("LOCAL_WORLD_SIZE", "1")) or not 0 <= local < visible:
        raise ValueError("CUDA_VISIBLE_DEVICES count must equal local torchrun processes")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        train(args, output, source_contract, source_step, rank, world, local, device)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def train(args, output, source_contract, source_step, rank, world, local, device):
    batch_size = args.global_batch_size // (world * args.grad_accum)
    torch.manual_seed(args.seed)
    output.mkdir(parents=True, exist_ok=True)

    from deploy.server_flow2act import load_policy_flow2act
    policy, contract = load_policy_flow2act(
        args.policy_checkpoint, device=device, processor_path=args.processor_path)
    policy.eval().requires_grad_(False)
    if policy.checkpoint_global_step != source_step or contract != source_contract:
        raise ValueError("Policy checkpoint changed between launch preflight and strict restoration")
    from mibot.data.datasets.libero_dataset_flow2act import LiberoDatasetFlow2Act
    from mibot.data.collate.custom_collate_flow2act import CustomCollateFlow2Act
    from flow2act.da3.da3_anyview_teacher import (
        DA3AnyViewTeacherFlow2Act, DA3_BACKBONE_FILENAME_FLOW2ACT, DA3_HEAD_FILENAME_FLOW2ACT)
    teacher_dir = Path(args.teacher_weights_dir).expanduser().resolve()
    teacher = DA3AnyViewTeacherFlow2Act(
        backbone_weights=teacher_dir / DA3_BACKBONE_FILENAME_FLOW2ACT,
        head_weights=teacher_dir / DA3_HEAD_FILENAME_FLOW2ACT, device=device)
    dataset = LiberoDatasetFlow2Act({"max_steps": 1, "train_datasets": {
        "root": args.data_root, "probe_only": True, "stage": "baseline", "action_length": 10,
        "execute_horizon": 10, "memory_length": 1, "augment": False, "batch_size": batch_size}})
    train_episodes, val_episodes = stratified_episode_split_flow2act(
        dataset.episodes, seed=args.seed, val_fraction=args.val_fraction)
    train_indices = episode_frame_indices_flow2act(train_episodes, dataset.sample_ends)
    val_indices = episode_frame_indices_flow2act(val_episodes, dataset.sample_ends)
    split = {name: [{k: dataset.episodes[i][k] for k in ("suite", "task_index", "episode_index")}
                    for i in indices] for name, indices in (("train", train_episodes), ("validation", val_episodes))}
    split.update(seed=args.seed, val_fraction=args.val_fraction)
    collate = CustomCollateFlow2Act(args.processor_path)
    train_subset = Subset(dataset, train_indices)
    sampler = DistributedSampler(train_subset, num_replicas=world, rank=rank, seed=args.seed,
                                 shuffle=True, drop_last=True)
    common = dict(batch_size=batch_size, num_workers=args.num_workers, collate_fn=collate,
                  pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_subset, sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(Subset(dataset, val_indices[rank::world]), shuffle=False, **common)
    fixed_indices = visual_samples(dataset, val_episodes)
    if not len(train_loader):
        raise ValueError("Train split is smaller than one microbatch")
    torch.manual_seed(args.seed)
    probe_cfg = Flow2ActDepthProbeConfig(input_dim=int(policy.vlm.config.text_config.hidden_size))
    probe = Flow2ActDepthProbe(probe_cfg).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_factor(step):
        if step < args.warmup_steps:
            return max(1, step + 1) / max(1, args.warmup_steps)
        progress = min(1.0, (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps))
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    training_config = {key: getattr(args, key) for key in (
        "global_batch_size", "grad_accum", "max_steps", "lr", "min_lr", "warmup_steps", "weight_decay",
        "clip_grad_norm", "si_log_weight", "gradient_weight", "log_depth_weight", "seed")}
    training_config.update(world_size=world, batch_per_gpu=batch_size,
                           data_root=str(Path(args.data_root).resolve()), teacher_weights_dir=str(teacher_dir),
                           processor_path=str(Path(args.processor_path).resolve()), decoder_trainable=False,
                           student_taps=list(policy.student_taps), teacher_taps=list(teacher.layer_ids))
    train_probe = DistributedDataParallel(probe, device_ids=[local]) if world > 1 else probe
    if rank == 0:
        write_json(output / "config_depth_probe_DepthOnly_flow2act.json",
                   {**vars(args), "probe": probe_cfg.to_dict(), "policy_contract": contract,
                    "policy_global_step": source_step, "training_config": training_config,
                    "kind": KIND, "batch_per_gpu": batch_size, "world_size": world})
        write_json(output / "episode_split_DepthOnly_flow2act.json", split)
        print(f"DepthOnly: frozen stage={policy.stage} policy_step={source_step}; output={output}", flush=True)
        print(f"GBS={args.global_batch_size} = GPUs{world} * microB{batch_size} * GA{args.grad_accum}; "
              f"train/val episodes={len(train_episodes)}/{len(val_episodes)}; only Adapter is optimized", flush=True)

    def forward(batch, training):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            encoded = policy.vlm.model(**{k: v for k, v in batch.items() if k in policy._QWEN_KEYS},
                                       use_cache=False, feature_taps=policy.student_taps)
            grids = tuple(policy._image_grids(batch, tap) for tap in encoded.hidden_states)
            target = encode_probe_teacher_depthonly_flow2act(teacher, batch["teacher_images"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = (train_probe if training else probe)(grids)
        decoded = teacher.decode_student(predicted, (18, 18), output_hw=(252, 252),
                                         use_checkpoint=training, batch_chunk_size=1)
        losses = depth_only_losses_flow2act(
            decoded["depth"], target["depth"], target["confidence"], si_log_weight=args.si_log_weight,
            gradient_weight=args.gradient_weight, log_depth_weight=args.log_depth_weight)
        return losses, decoded["depth"], target, batch

    def policy_action():
        batch = collate([dataset[fixed_indices[0]]])
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        inputs = {key: value for key, value in batch.items() if key in policy._QWEN_KEYS}
        inputs["state"] = batch["state"]
        n = policy.memory_length - 1
        inputs.update(history_inputs=[dict(inputs) for _ in range(n)],
                      history_state=torch.zeros((1, n, 1, 32), device=device, dtype=torch.bfloat16),
                      history_valid=torch.zeros((1, n), device=device, dtype=torch.bool),
                      history_ages=torch.zeros((1, n), device=device, dtype=torch.int64),
                      action=torch.zeros((1, policy.action_shape[0], 32), device=device, dtype=torch.bfloat16))
        inputs["action_mask"] = torch.zeros_like(inputs["action"], dtype=torch.int32)
        inputs["action_mask"][..., :7] = 1
        with torch.random.fork_rng(devices=[local]), torch.no_grad():
            torch.manual_seed(args.seed + 123)
            return policy.generate(inputs).detach().cpu()

    before_action = policy_action() if rank == 0 and args.verify_frozen_policy else None

    def validate(step):
        probe.eval()
        sums = torch.zeros(len(LOSS_KEYS) + 1, device=device, dtype=torch.float64)
        metric_sums = depth_metric_sums_flow2act(
            torch.ones((1, 2, 5, 5), device=device), torch.ones((1, 2, 5, 5), device=device),
            torch.zeros((1, 2, 5, 5), device=device))
        # Start zero counts, including the placeholder's total-pixel count.
        metric_sums = {key: value * 0 for key, value in metric_sums.items()}
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if args.val_max_batches and index >= args.val_max_batches:
                    break
                losses, depth, target, moved = forward(batch, False)
                size = moved["input_ids"].shape[0]
                sums += torch.stack([losses[key].double() * size for key in LOSS_KEYS] +
                                    [torch.tensor(size, device=device, dtype=torch.float64)])
                batch_metrics = depth_metric_sums_flow2act(depth, target["depth"], target["confidence"])
                for key, value in batch_metrics.items():
                    metric_sums[key] += value
        if world > 1:
            dist.all_reduce(sums)
            packed = torch.stack(list(metric_sums.values()))
            dist.all_reduce(packed)
            metric_sums = dict(zip(metric_sums, packed))
        metrics = {key: float(value / sums[-1].clamp_min(1)) for key, value in zip(LOSS_KEYS, sums[:-1])}
        metrics.update(finalize_depth_metrics_flow2act(metric_sums))
        metrics.update(step=step, frames=int(sums[-1]), max_batches_per_rank=args.val_max_batches,
                       reference="DA3 pseudo-depth consistency; not true metric-depth accuracy")
        if rank == 0:
            with (output / "validation_metrics_DepthOnly_flow2act.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            print("validation " + json.dumps(metrics, allow_nan=False), flush=True)
            with torch.no_grad():
                for index in fixed_indices:
                    batch = collate([dataset[index]])
                    _, depth, target, moved = forward(batch, False)
                    save_depths(output, step, depth, target["depth"], target["confidence"], moved["teacher_images"],
                                suite=batch["suite"][0], episode=int(batch["episode_index"][0]), frame=int(batch["frame_index"][0]))
        probe.train()

    def save(step):
        if rank == 0:
            destination = output / f"probe_step_{step:07d}_DepthOnly_flow2act.pt"
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            torch.save(dict(kind=KIND, step=step, probe=probe.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), probe_config=probe_cfg.to_dict(),
                            policy_contract=contract, policy_checkpoint=str(Path(args.policy_checkpoint).resolve()),
                            policy_global_step=source_step, training_config=training_config, split=split),
                       temporary)
            temporary.replace(destination)

    epoch = 0
    sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    probe.train()
    gradient_audited = False
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros(len(LOSS_KEYS), device=device)
        valid_evidence = torch.zeros((), device=device, dtype=torch.int64)
        for micro in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            sync = train_probe.no_sync() if world > 1 and micro + 1 < args.grad_accum else nullcontext()
            with sync:
                losses, _, target, _ = forward(batch, True)
                if not gradient_audited:
                    valid_evidence += (torch.isfinite(target["depth"]) & (target["depth"] > 1e-6)
                                       & torch.isfinite(target["confidence"]) & (target["confidence"] > 1)).any().long()
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("Non-finite DepthOnly loss")
                (losses["loss"] / args.grad_accum).backward()
                total += torch.stack([losses[key].detach() for key in LOSS_KEYS]) / args.grad_accum
        if not gradient_audited:
            if world > 1:
                dist.all_reduce(valid_evidence)
            frozen_ok = all(value.grad is None for value in policy.parameters())
            frozen_ok &= all(value.grad is None for value in teacher.backbone.parameters())
            frozen_ok &= all(value.grad is None for value in teacher.head.parameters())
            named = dict(probe.named_parameters())
            names = ["input_norm.weight", "channel.weight", "refine1.weight", "refine2.weight"]
            names += [f"heads.{index}.weight" for index in range(4)]
            audit = {name: dict(finite=bool(torch.isfinite(named[name].grad).all()) if named[name].grad is not None else False,
                               gradient_l1=float(named[name].grad.float().abs().sum()) if named[name].grad is not None else 0.0)
                     for name in names}
            has_evidence = bool(valid_evidence > 0)
            if not frozen_ok or any(not value["finite"] or (has_evidence and value["gradient_l1"] <= 0) for value in audit.values()):
                raise RuntimeError(f"DepthOnly gradient isolation failed: frozen={frozen_ok}, adapter={audit}")
            write_json(output / f"gradient_audit_rank{rank}_DepthOnly_flow2act.json",
                       dict(step=step, policy_and_teacher_have_no_gradients=frozen_ok, adapter=audit,
                            nonzero_gradient_check=has_evidence))
            gradient_audited = has_evidence
        grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), args.clip_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0:
            if world > 1:
                dist.all_reduce(total)
                total /= world
            if rank == 0:
                row = dict(step=step, max_steps=args.max_steps, lr=scheduler.get_last_lr()[0],
                           grad_norm=float(grad_norm), **{key: float(value) for key, value in zip(LOSS_KEYS, total)})
                with (output / "train_metrics_DepthOnly_flow2act.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"Adapter updates {step}/{args.max_steps}: " + json.dumps(row, allow_nan=False), flush=True)
        if (args.val_every and step % args.val_every == 0) or step == args.max_steps:
            validate(step)
        if (args.save_every and step % args.save_every == 0) or step == args.max_steps:
            save(step)
    if before_action is not None:
        after_action = policy_action()
        torch.testing.assert_close(before_action, after_action, rtol=0, atol=0)
        write_json(output / "policy_action_invariance_DepthOnly_flow2act.json",
                   dict(identical=True, max_absolute_error=float((before_action - after_action).abs().max()),
                        policy_global_step=source_step, probe_final_step=args.max_steps))
    write_json(output / f"completed_rank{rank}_DepthOnly_flow2act.json",
               dict(kind=KIND, policy_global_step=source_step,
                    final_step=args.max_steps, optimizer_updates=args.max_steps,
                    microbatches=args.max_steps * args.grad_accum,
                    global_batch_size=args.global_batch_size, batch_per_gpu=batch_size,
                    gradient_accumulation=args.grad_accum, world_size=world,
                    adapter_parameters=sum(parameter.numel() for parameter in probe.parameters()),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    if rank == 0:
        print(f"DepthOnly complete: adapter updates={args.max_steps}; source policy unchanged", flush=True)


if __name__ == "__main__":
    os.umask(0)
    main()
