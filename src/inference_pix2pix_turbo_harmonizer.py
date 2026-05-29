# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference for the multi-view fixer model. Loads a legacy
diffusion-fixer .pkl checkpoint (state_dict_unet / state_dict_vae) and runs the
Cosmos-predict2 pipeline image-by-image, writing the outputs to a sibling
folder named ``<input>_<model_identifier>``.

Supports autoregressive temporal conditioning: each frame is fed alongside
previously-predicted output frames (selected by ``--offset_list``) so the model
can exploit temporal context. For the first frames where not enough prior
outputs exist, falls back to a non-temporal pass (V=1) with the same model.
Pass ``--nontemporal`` to disable temporal conditioning entirely.
"""

import argparse
import os
from glob import glob

import imageio
import numpy as np
import torch
from einops import rearrange
from natsort import natsorted
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from pix2pix_turbo_harmonizer import Pix2Pix_Turbo


def save_folder2video(save_dir):
    video_file = os.path.join(save_dir, save_dir.split("/")[-1] + "_video.mp4")
    im_files = natsorted(glob(os.path.join(save_dir, "*.png")) + glob(os.path.join(save_dir, "*.jpg")))
    ims = [imageio.v2.imread(f) for f in im_files]
    for i in range(len(ims)):
        if ims[i].shape[0] % 2 == 1:
            ims[i] = ims[i][:-1, :, :]
        if ims[i].shape[1] % 2 == 1:
            ims[i] = ims[i][:, :-1, :]
    imageio.v2.mimwrite(video_file, ims, fps=30, macro_block_size=1)


def extend_path(orig_path, suffix):
    orig_path = orig_path.rstrip("/")
    parent, last = os.path.split(orig_path)
    return os.path.join(parent, f"{last}_{suffix}")


RESOLUTION_MAP = {
    1024: (1024, 576),
    960: (960, 544),
    1360: (1360, 768),
}


def load_image_tensor(path, size, device, dtype):
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    t = transforms.ToTensor()(img)
    t = transforms.Normalize([0.5], [0.5])(t).unsqueeze(0).to(device=device, dtype=dtype)
    return t  # (1, 3, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="directory of input images")
    parser.add_argument("--model_path", type=str, required=True, help="path to the .pkl checkpoint")
    parser.add_argument("--model_identifier", type=str, default="fixer", help="suffix for the output folder")
    parser.add_argument("--timestep", type=int, default=250)
    parser.add_argument("--resolution", type=int, default=1024, choices=list(RESOLUTION_MAP.keys()))
    parser.add_argument("--vae_skip_connection", action="store_true")
    parser.add_argument("--use_sched", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--max_frames", type=int, default=3_000_000)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument(
        "--offset_list",
        type=int,
        nargs="+",
        default=[-1, -2, -3, -4],
        help="negative offsets into the list of previously-predicted outputs used as temporal references",
    )
    parser.add_argument(
        "--nontemporal",
        action="store_true",
        help="disable temporal conditioning entirely; run frame-by-frame with V=1",
    )
    args = parser.parse_args()

    assert all(o < 0 for o in args.offset_list), "offsets must be negative (refer to past frames)"

    args.output_dir = extend_path(args.input_image, args.model_identifier)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = Pix2Pix_Turbo(
        pretrained_path=args.model_path,
        timestep=args.timestep,
        train_full_unet=True,
        freeze_vae=False,
        vae_skip_connection=args.vae_skip_connection,
        use_sched=args.use_sched,
        device=device,
        dtype=dtype,
    )
    model.set_eval()
    model = model.to(device=device, dtype=dtype)

    size = RESOLUTION_MAP[args.resolution]
    n_refs = len(args.offset_list)
    min_history = -min(args.offset_list)  # smallest history length that makes every offset valid

    img_paths = sorted(
        glob(os.path.join(args.input_image, "*.png"))
        + glob(os.path.join(args.input_image, "*.jpg"))
        + glob(os.path.join(args.input_image, "*.jpeg"))
    )
    img_paths = img_paths[: args.max_frames][args.start_frame :: args.skip_frames]

    cond_frame_paths = []  # autoregressive: previously written outputs

    for img_path in tqdm(img_paths):
        input_image = Image.open(img_path).convert("RGB")
        original_shape = input_image.size
        input_image_resized = input_image.resize(size, Image.BILINEAR)

        c_t = transforms.ToTensor()(input_image_resized)
        c_t = transforms.Normalize([0.5], [0.5])(c_t).unsqueeze(0).to(device=device, dtype=dtype)  # (1, 3, H, W)

        have_history = (not args.nontemporal) and len(cond_frame_paths) >= min_history

        if have_history:
            ref_paths = [cond_frame_paths[off] for off in args.offset_list]
            ref_tensors = [load_image_tensor(p, size, device, dtype) for p in ref_paths]
            stacked = torch.stack([c_t] + ref_tensors, dim=1)  # (1, 1+n_refs, 3, H, W)
            stacked = rearrange(stacked, "b v c h w -> b c v h w")
        else:
            stacked = rearrange(c_t.unsqueeze(1), "b v c h w -> b c v h w")  # (1, 3, 1, H, W)

        with torch.no_grad():
            output = model(stacked).float()
        output = rearrange(output, "b c v h w -> b v c h w")
        output_image = output[0, 0].cpu() * 0.5 + 0.5
        output_image = torch.clamp(output_image, 0.0, 1.0)
        output_pil = transforms.ToPILImage()(output_image).resize(original_shape)

        save_path = os.path.join(args.output_dir, os.path.basename(img_path))
        output_pil.save(save_path)
        cond_frame_paths.append(save_path)

    if args.save_video:
        save_folder2video(args.output_dir)


if __name__ == "__main__":
    main()
