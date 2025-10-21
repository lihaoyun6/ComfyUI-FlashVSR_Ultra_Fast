#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import math
import torch
import folder_paths
import comfy.utils

import numpy as np
import torch.nn.functional as F

from einops import rearrange
from .src.diffsynth import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline
from .src.utils.utils import Buffer_LQ4x_Proj
from .src.utils.TCDecoder import build_tcdecoder

def vram_clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_device_list():
    devs = ["auto"]
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except Exception:
        pass
    return devs+["cpu"]

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    
    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]

    return cropped_tensor.squeeze(0)

def prepare_input_tensor(image_tensor: torch.Tensor, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx]
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        tensor_out = tensor_out.to(dtype)
        frames.append(tensor_out)

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    vram_clean()
    
    return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def init_pipeline(mode, device):
    model_path = os.path.join(folder_paths.models_dir, "FlashVSR")
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist!\nPlease save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!\nPlease save it to "{model_path}"')
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_path):
        raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist!\nPlease save it to "{model_path}"')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "src", "utils", "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path), strict=False)
        
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
    if os.path.exists(lq_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    
    return pipe

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing"):
        self.desc = desc
        self.pbar = None
        self.iterable = None
        self.total = total
        
        if iterable is not None:
            try:
                self.total = len(iterable)
                self.iterable = iter(iterable)
            except TypeError:
                if self.total is None:
                    raise ValueError("Total must be provided for iterables with no length.")

        elif self.total is not None:
            pass
            
        else:
            raise ValueError("Either iterable or total must be provided.")
            
    def __iter__(self):
        if self.iterable is None:
            raise TypeError(f"'{type(self).__name__}' object is not iterable. Did you mean to use it with a 'with' statement?")
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self
    
    def __next__(self):
        if self.iterable is None:
            raise TypeError("Cannot call __next__ on a non-iterable cqdm object.")
        try:
            val = next(self.iterable)
            if self.pbar:
                self.pbar.update(1)
            return val
        except StopIteration:
            raise
            
    def __enter__(self):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __len__(self):
        return self.total

class FlashVSRNode:
    @classmethod
    def INPUT_TYPES(cls):
        device_choices = get_device_list()
            
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Sequential video frames as IMAGE tensor batch"
                }),
                "mode": (["tiny", "full"], {
                    "default": "tiny",
                }),
                "scale": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 4,
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable tiling: faster decode but higher VRAM usage.\nSet to True for lower memory consumption at the cost of speed."
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable tiling: faster decode but higher VRAM usage.\nSet to True for lower memory consumption at the cost of speed."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Significantly reduces VRAM usage at the cost of speed."
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                }),
                "tile_overlap": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload DiT before decoding to reduce VRAM peak at the cost of speed."
                }),
                "sparse_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Recommended: 1.5 or 2.0\n1.5 → faster; 2.0 → more stable"
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Recommended: 1.0 to 3.0\n1.0 → less vram; 3.0 → high quality"
                }),
                "local_range": ("INT", {
                    "default": 11,
                    "min": 9,
                    "max": 11,
                    "step": 2,
                    "tooltip": "Recommended: 9 or 11\nlocal_range=9 → sharper details; 11 → more stable results"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624
                }),
                "device": (device_choices, {
                    "default": device_choices[0],
                    "tooltip": "Device to load the weights, default: auto (CUDA if available, else CPU)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Download the entire "FlashVSR" folder all the files inside it from "https://huggingface.co/JunhaoZhuang/FlashVSR" and put it in the "ComfyUI/models"'
    
    def main(self, frames, mode, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, device):
        if device == "auto":
            _device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if tiled_dit and (tile_overlap > tile_size / 2):
            raise ValueError('"tile_overlap" must not be greater than half of "tile_size"')
        
        dtype = torch.bfloat16
        pipe = init_pipeline(mode, _device)
        
        if tiled_dit:
            N, H, W, C = frames.shape
            num_aligned_frames = largest_8n1_leq(N + 4) - 4
            
            final_output_canvas = torch.zeros(
                (num_aligned_frames, H * scale, W * scale, C), 
                dtype=torch.float32, 
                device="cpu"
            )
            weight_sum_canvas = torch.zeros_like(final_output_canvas)
            tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
            latent_tiles_cpu = []
            
            for i, (x1, y1, x2, y2) in enumerate(cqdm(tile_coords, desc="Processing Tiles")):
                print(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: coords ({x1},{y1}) to ({x2},{y2})")
                input_tile = frames[:, y1:y2, x1:x2, :]
                
                _tile = input_tile.to(_device)
                LQ_tile, th, tw, F = prepare_input_tensor(_tile, scale=scale, dtype=torch.bfloat16)
                del _tile
                vram_clean()
                
                output_tile_gpu = pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                    LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                    topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                    color_fix=color_fix, unload_dit=unload_dit
                )
                
                processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
                
                mask_nchw = create_feather_mask(
                    (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                    tile_overlap * scale
                ).to("cpu")
                mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
                out_x1, out_y1 = x1 * scale, y1 * scale
                
                tile_H_scaled = processed_tile_cpu.shape[1]
                tile_W_scaled = processed_tile_cpu.shape[2]
                out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
                final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
                weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
                
                del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
                vram_clean()
                
            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            final_output = final_output_canvas / weight_sum_canvas
        else:
            _frames = frames.to(_device)
            LQ, th, tw, F = prepare_input_tensor(_frames, scale=scale, dtype=dtype)
            
            video = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                progress_bar_cmd=cqdm, LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix = color_fix, unload_dit=unload_dit
            )
            
            final_output = tensor2video(video)
            
            del pipe, video, LQ
            vram_clean()
        
        print("[FlashVSR] Inference complete. Cleaning up models and cache...")
        return (final_output,)

NODE_CLASS_MAPPINGS = {
    "FlashVSRNode": FlashVSRNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRNode": "FlashVSR",
}