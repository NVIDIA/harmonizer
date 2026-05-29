# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import requests
from diffusers import DPMSolverMultistepScheduler
from tqdm import tqdm

from .pipeline import FastDenoiser, FastTokenizer, FastUnconditioner
from .text2image_dit import MiniTrainDIT


# +
def make_1step_sched_base():
    noise_scheduler_1step = DPMSolverMultistepScheduler.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers", subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1)
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod
    return noise_scheduler_1step


def make_1step_sched_sprint():
    noise_scheduler_1step = DPMSolverMultistepScheduler.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers", subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1)
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod
    return noise_scheduler_1step


def make_1step_sched(hf_path):
    noise_scheduler_1step = DPMSolverMultistepScheduler.from_pretrained(hf_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1)
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod
    return noise_scheduler_1step


# -


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(outf, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


__all__ = [
    "FastDenoiser",
    "FastTokenizer",
    "FastUnconditioner",
    "MiniTrainDIT",
    "download_url",
    "make_1step_sched",
    "make_1step_sched_base",
    "make_1step_sched_sprint",
    "my_vae_decoder_fwd",
    "my_vae_encoder_fwd",
]
