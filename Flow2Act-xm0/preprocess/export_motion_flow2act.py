"""Export main-camera current-anchor native-UV trajectories on canonical contexts.

No raw XYZ / hidden-feature motion labels are read or generated.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flow2act.cache import (
    FILES_FLOW2ACT,SCHEMA_FLOW2ACT,canonical_suite_flow2act)
from flow2act.geometry import anchor_grid_flow2act,lift_trajectories_flow2act
from flow2act.track4world_export import (
    NativeUVTeacherFlow2Act,canonical_windows_flow2act,
    gather_frames_flow2act,self_test_flow2act)


def read_main_video_flow2act(path,expected_length,ffmpeg_bin=None):
    try:
        import av
    except ImportError:
        import shutil,subprocess
        binary=ffmpeg_bin or os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
        if not binary:
            try:
                import imageio_ffmpeg
                binary=imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError as error:
                raise RuntimeError("Install minimal av or provide an AV1-capable --ffmpeg-bin") from error
        result=subprocess.run([str(binary),"-v","error","-i",str(path),"-f","rawvideo",
                               "-pix_fmt","rgb24","pipe:1"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        frame_bytes=256*256*3
        if len(result.stdout)!=expected_length*frame_bytes:
            raise ValueError(f"Exact ffmpeg byte count mismatch: {len(result.stdout)} != {expected_length*frame_bytes}")
        values=np.frombuffer(result.stdout,dtype=np.uint8).reshape(expected_length,256,256,3).copy()
        print(f"PyAV unavailable; exact AV1 ffmpeg decode: {expected_length} frames, no FPS conversion",flush=True)
        return torch.from_numpy(values).permute(0,3,1,2).contiguous()
    frames=[]
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb=frame.to_ndarray(format="rgb24")
            if rgb.shape!=(256,256,3):
                raise ValueError(f"Expected full-FOV 256x256 source video: {path}: {rgb.shape}")
            frames.append(rgb)
    if len(frames)!=expected_length:
        raise ValueError(f"Video/metadata frame count mismatch: {len(frames)} != {expected_length}")
    return torch.from_numpy(np.stack(frames)).permute(0,3,1,2).contiguous()


def selected_episodes_flow2act(data_root,suites,episode_index=None,limit=None):
    selected=[]
    for suite in suites:
        suite=canonical_suite_flow2act(suite)
        source=Path(data_root)/(suite+"_no_noops_1.0.0_lerobot")
        info=json.loads((source/"meta/info.json").read_text())
        if info["codebase_version"]!="v2.1":raise ValueError("LeRobot v2.1 required")
        first,last=map(int,info["splits"]["train"].split(":"))
        rows=[json.loads(line) for line in (source/"meta/episodes.jsonl").read_text().splitlines()]
        count=0
        for row in sorted(rows,key=lambda x:int(x["episode_index"])):
            index=int(row["episode_index"])
            if not first<=index<last or episode_index is not None and index!=episode_index:continue
            fields={"episode_index":index,"episode_chunk":index//int(info["chunks_size"]),
                    "video_key":"observation.images.image"}
            selected.append((suite,index,int(row["length"]),source/info["video_path"].format(**fields)))
            count+=1
            if limit is not None and count>=limit:break
    if not selected:raise ValueError("No selected train episodes")
    return selected


def cache_metadata_flow2act(motion_3d_enabled):
    return {"schema":SCHEMA_FLOW2ACT,"horizon":50,"image_hw":[256,256],"grid_hw":[32,32],
        "has_motion_3d":bool(motion_3d_enabled),"coordinate":"displacement_xy_div_wh",
        "source_camera":"observation.images.image","canonical_context_frames":128,"anchor_ownership":78,
        "decoder_sequence_length":16,"decoder_stride":8,"decoder_iterations":4,
        "decoder_temporal_attention":True,"decoder_overlap_warm_start":True,
        "geometry_batch":1,"geometry_real_frames_only":True,"decoder_source_self_slot":True,
        "visibility":"native_sigmoid_visibility_and_confidence","endpoint_bounds_before_float16":True,
        "optional_3d":"native_UV_plus_original_DA3_depth_fullK_C2W_in_current_camera_div_anchor_median_depth",
        "raw_xyz_flow_used":False}


def _open_arrays_flow2act(directory,length,motion_3d):
    specs={"motion_2d":((length,50,2,32,32),np.float16),
        "visconf":((length,50,2,32,32),np.float16),"in_bounds":((length,50,32,32),np.bool_),
        "horizon_valid":((length,50),np.bool_)}
    if motion_3d:
        specs.update(motion_3d=((length,50,3,32,32),np.float16),
                     valid_3d=((length,50,32,32),np.bool_),anchor_scale=((length,),np.float32))
    arrays={}
    for name,(shape,dtype) in specs.items():
        filename=FILES_FLOW2ACT[name]+(".partial" if name=="horizon_valid" else "")
        arrays[name]=np.lib.format.open_memmap(directory/filename,mode="w+",dtype=dtype,shape=shape)
        arrays[name][:]=0
    return arrays


def export_episode_flow2act(teacher,frames,output_dir,motion_3d=False):
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True,exist_ok=True);output_dir.parent.chmod(0o777)
    # EFS supports file replacement but not directory rename. Reserve the final
    # directory exclusively; publish root metadata only after all episodes finish.
    output_dir.mkdir(exist_ok=False);output_dir.chmod(0o777)
    arrays=_open_arrays_flow2act(output_dir,len(frames),motion_3d)
    anchor=anchor_grid_flow2act().permute(2,0,1).cpu()
    for start,stop,owned_stop in canonical_windows_flow2act(len(frames)):
        features,geometry=teacher.extract_context(frames[start:stop])
        if motion_3d:
            np.savez_compressed(output_dir/f"native_DA3_context_{start:06d}_flow2act.npz",
                **{key:value.float().cpu().numpy() for key,value in geometry.items()},
                frame_indices=np.arange(start,stop,dtype=np.int64))
        for global_anchor in range(start,owned_stop):
            local_anchor=global_anchor-start
            flow,probability,horizon_valid=teacher.decode_anchor(features,local_anchor,stop-start,iters=4)
            flow=flow.float().cpu();probability=probability.float().cpu()
            endpoint=anchor[None]+flow
            bounds=torch.isfinite(flow).all(1)&torch.isfinite(endpoint).all(1)
            bounds&=(endpoint[:,0]>=0)&(endpoint[:,0]<=255)&(endpoint[:,1]>=0)&(endpoint[:,1]<=255)
            bounds&=torch.as_tensor(horizon_valid)[:,None,None]
            safe=torch.where(bounds[:,None],flow,torch.zeros_like(flow))
            arrays["motion_2d"][global_anchor]=(safe/256.).numpy().astype(np.float16)
            arrays["visconf"][global_anchor]=probability.numpy().astype(np.float16)
            arrays["in_bounds"][global_anchor]=bounds.numpy()
            arrays["horizon_valid"][global_anchor]=horizon_valid
            if motion_3d:
                ids,_=gather_frames_flow2act(local_anchor,stop-start)
                index=torch.as_tensor(ids,device=geometry["depth"].device,dtype=torch.long)
                lifted=lift_trajectories_flow2act(
                    safe.to(index.device),geometry["depth"].index_select(0,index),
                    geometry["intrinsics"].index_select(0,index),geometry["camera_to_world"].index_select(0,index),
                    geometry["depth_confidence"].index_select(0,index),image_hw=(256,256))
                valid3=lifted["valid_3d"]&bounds.to(index.device)
                value=torch.where(valid3[:,None],lifted["motion_3d"],0.)
                arrays["motion_3d"][global_anchor]=value.cpu().numpy().astype(np.float16)
                arrays["valid_3d"][global_anchor]=valid3.cpu().numpy()
                arrays["anchor_scale"][global_anchor]=float(lifted["anchor_scale"])
        del features,geometry
    for value in arrays.values():value.flush()
    arrays.clear()
    for child in output_dir.iterdir():child.chmod(0o777)
    # This required scientific array is the last publication. An interrupted
    # episode lacks its required filename, so existing-root readers fail fast.
    horizon_path=output_dir/FILES_FLOW2ACT["horizon_valid"]
    horizon_path.with_name(horizon_path.name+".partial").replace(horizon_path)
    horizon_path.chmod(0o777)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=ROOT/"data/libero_lerobot")
    parser.add_argument("--output-root",type=Path,default=ROOT/"checkpoints/cache/libero4mix_motion3d_flow2act")
    parser.add_argument("--track4world-root",type=Path,default=ROOT/"third_party/Track4World")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--ffmpeg-bin",default=os.environ.get("FFMPEG_BIN"))
    parser.add_argument("--suites",nargs="+",default=["libero_10","libero_goal","libero_object","libero_spatial"])
    parser.add_argument("--episode-index",type=int)
    parser.add_argument("--limit-episodes",type=int)
    parser.add_argument("--motion-3d-enabled",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--smoke",action="store_true",help="One anchor diagnostic only; never publishes a training cache")
    parser.add_argument("--smoke-context-frames",type=int,default=128)
    args=parser.parse_args()
    args.output_root=args.output_root.expanduser().resolve()
    if args.self_test:self_test_flow2act();return
    episodes=selected_episodes_flow2act(args.data_root,args.suites,args.episode_index,args.limit_episodes)
    metadata=cache_metadata_flow2act(args.motion_3d_enabled)
    header=args.output_root/"metadata_flow2act.json"
    if header.exists() and json.loads(header.read_text())!=metadata:
        raise ValueError("Existing cache scientific contract differs; use a new output root")
    teacher=NativeUVTeacherFlow2Act(args.track4world_root,args.checkpoint,args.device)
    for suite,index,length,video in episodes:
        frames=read_main_video_flow2act(video,length,args.ffmpeg_bin)
        if args.smoke:
            count=min(length,args.smoke_context_frames)
            if not 1<=count<=128:raise ValueError("smoke context must contain 1..128 real frames")
            features,geometry=teacher.extract_context(frames[:count])
            flow,prob,valid=teacher.decode_anchor(features,0,count)
            destination=args.output_root/"smoke"
            destination.mkdir(parents=True,exist_ok=True);destination.chmod(0o777)
            path=destination/"native_UV_smoke_flow2act.npz"
            diagnostics={key:value.float().cpu().numpy() for key,value in geometry.items()}
            if args.motion_3d_enabled:
                ids,_=gather_frames_flow2act(0,count)
                index=torch.as_tensor(ids,device=geometry["depth"].device,dtype=torch.long)
                lifted=lift_trajectories_flow2act(
                    flow.to(index.device),geometry["depth"].index_select(0,index),
                    geometry["intrinsics"].index_select(0,index),geometry["camera_to_world"].index_select(0,index),
                    geometry["depth_confidence"].index_select(0,index),image_hw=(256,256))
                anchor=anchor_grid_flow2act().permute(2,0,1).to(index.device)
                endpoint=anchor[None]+flow.to(index.device)
                bounds=torch.isfinite(endpoint).all(1)
                bounds&=(endpoint[:,0]>=0)&(endpoint[:,0]<=255)&(endpoint[:,1]>=0)&(endpoint[:,1]<=255)
                valid3=lifted["valid_3d"]&bounds&torch.as_tensor(valid,device=index.device)[:,None,None]
                diagnostics.update(
                    motion_3d=torch.where(valid3[:,None],lifted["motion_3d"],0.).cpu().numpy(),
                    valid_3d=valid3.cpu().numpy(),anchor_scale=lifted["anchor_scale"].cpu().numpy())
            np.savez_compressed(path,displacement_pixels=flow.numpy(),visconf=prob.numpy(),horizon_valid=valid,
                **diagnostics)
            path.chmod(0o777)
            print(f"Smoke only, no training cache published: context={count}, valid_horizons={valid.sum()}, {path}")
            return
        export_episode_flow2act(teacher,frames,args.output_root/suite/f"episode_{index:06d}",args.motion_3d_enabled)
        print(f"Exported {suite} episode {index:06d}: {length} current anchors",flush=True)
    args.output_root.mkdir(parents=True,exist_ok=True);args.output_root.chmod(0o777)
    if not header.exists():
        temporary=header.with_name("metadata_flow2act.tmp")
        temporary.write_text(json.dumps(metadata,indent=2)+"\n");temporary.chmod(0o777)
        temporary.replace(header);header.chmod(0o777)


if __name__=="__main__":main()
