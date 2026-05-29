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

import torch
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.configs.base.config_text2image import (
    Text2ImagePipelineConfig,
    get_cosmos_predict2_text2image_pipeline,
)
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from cosmos_predict2.tokenizers.tokenizer import CausalConv3d, ResidualBlock
from imaginaire.lazy_config import LazyDict

from model import FastDenoiser, FastTokenizer, FastUnconditioner, MiniTrainDIT


def get_pipeline_config(
    base_dir: str = "/work/models/base", fast_pipeline: bool = True
) -> LazyDict[Text2ImagePipelineConfig]:
    config = get_cosmos_predict2_text2image_pipeline(model_size="0.6B", fast_tokenizer=True)
    ### MiniTrainDIT
    config.dit_path = f"{base_dir}/model_fast_tokenizer.pt"
    config.tokenizer["vae_pth"] = f"{base_dir}/tokenizer_fast.pth"
    config.guardrail_config.enabled = False
    if fast_pipeline:
        config.net["_target_"] = MiniTrainDIT
    config.net.atten_backend = "torch"
    return config


def load_ckpt_from_state_dict(net_pix2pix, net_disc, optimizer, optimizer_disc, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")

    net_pix2pix.unet.load_state_dict(sd["state_dict_unet"])
    net_pix2pix.vae.load_state_dict(sd["state_dict_vae"])

    net_disc.load_state_dict({k[7:]: v for k, v in sd["net_disc"].items()})

    optimizer.load_state_dict(sd["optimizer"])
    optimizer_disc.load_state_dict(sd["optimizer_disc"])

    print()
    print("!!!! loading, load pretrained weight from", pretrained_path)
    print()
    return net_pix2pix, net_disc, optimizer, optimizer_disc


def save_ckpt(net_pix2pix, net_disc, optimizer, optimizer_disc, outf, train_full_unet=False, freeze_vae=False):
    sd = {}
    sd["state_dict_unet"] = net_pix2pix.unet.state_dict()
    sd["state_dict_vae"] = net_pix2pix.vae.state_dict()
    sd["net_disc"] = net_disc.state_dict()
    sd["optimizer"] = optimizer.state_dict()
    sd["optimizer_disc"] = optimizer_disc.state_dict()
    torch.save(sd, outf)


CACHE_T = 2


def my_vae_encoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    # downsamples
    l_blocks = []
    for layer in self.downsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
        l_blocks.append(x)

    # middle
    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    # head
    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)

    self.current_down_blocks = l_blocks
    return x


def my_vae_decoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    # conv1
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    skip_convs = [
        self.skip_conv_0,
        self.skip_conv_1,
        self.skip_conv_2,
        self.skip_conv_3,
        self.skip_conv_4,
        self.skip_conv_5,
        self.skip_conv_6,
        self.skip_conv_7,
        self.skip_conv_8,
    ]
    skip_acts = self.incoming_skip_acts

    # middle
    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    enc_dec_mapping = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4, 8: 6, 10: 7, 12: 9, 14: 10}

    for dec_idx, layer in enumerate(self.upsamples):
        if dec_idx in enc_dec_mapping:
            enc_indx = enc_dec_mapping[dec_idx]
            layer_index = list(enc_dec_mapping.keys()).index(dec_idx)

            skip_input = skip_acts[::-1][enc_indx]
            skip_in = skip_convs[layer_index](skip_input)  # 1x1 conv
            x = x + skip_in  # add skip

        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    # head
    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x


class AOTModule(torch.nn.Module):
    """
    AOTModule is a wrapper for the AOT-compiled module.
    """

    def __init__(self, filename: str):
        super().__init__()
        self.mod = torch._inductor.aoti_load_package(filename)

    def forward(self, *args, **kwargs):
        return self.mod(*args, **kwargs)


class ImageEncoder(torch.nn.Module):
    def __init__(self, tokenizer: FastTokenizer):
        super().__init__()
        self.fn = lambda x: tokenizer.encode(x[:, :, None, :, :])

    def forward(self, x):
        return self.fn(x)


class ImageDecoder(torch.nn.Module):
    def __init__(self, tokenizer: FastTokenizer):
        super().__init__()
        self.fn = lambda z_denoised: tokenizer.decode(z_denoised)[:, :, 0]

    def forward(self, z_denoised):
        return self.fn(z_denoised)


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(
        self,
        experiment_name=None,
        s3_checkpoint_dir=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        lora_rank_unet=8,
        lora_rank_vae=4,
        hf_path=None,
        unet_in_channels=4,
        freeze_vae_encoder=False,
        freeze_vae=False,
        train_full_unet=True,
        timestep=999,
        use_sched=False,
        vae_skip_connection=False,
        batch_size=1,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        inference_only_mode=False,
    ):
        super().__init__()

        self.experiment_name = experiment_name
        self.s3_checkpoint_dir = s3_checkpoint_dir
        self.batch_size = batch_size

        self.timesteps = torch.tensor([timestep], device=device)
        self.timesteps = torch.cat([self.timesteps] * batch_size, 0).to(dtype=dtype)
        self.timesteps_int = timestep

        self.train_full_unet = train_full_unet
        self.freeze_vae = freeze_vae
        self.freeze_vae_encoder = freeze_vae_encoder

        self.use_sched = use_sched
        if self.use_sched:
            self.sched = None  # make_1step_sched()

        # Create Cosmos model
        self.initialize_cosmos_model(
            inference_only_mode=inference_only_mode,
            vae_skip_connection=vae_skip_connection,
            device=device,
            dtype=dtype,
        )
        if self.inference_only_mode:
            self.set_eval()
        else:
            self.set_train()

        # Load pretrained weights if provided
        if pretrained_path is not None:
            print("loading from", pretrained_path)
            sd = torch.load(pretrained_path, map_location="cpu")
            self.unet.load_state_dict(sd["state_dict_unet"], strict=False)
            self.vae.load_state_dict(sd["state_dict_vae"], strict=False)

        # Because sigma and conditioning are constant, we can compute them ahead of time.
        if self.inference_only_mode:
            self.unet.compute_caching(sigma=self.timesteps / 1000, condition=self.condition)

        # Log number of trainable parameters
        print("=" * 50)
        print(
            f"Number of trainable parameters in UNet: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print(
            f"Number of trainable parameters in VAE: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print("=" * 50)

    def initialize_cosmos_model_training(
        self, config: LazyDict[Text2ImagePipelineConfig], device: str | torch.device, dtype: torch.dtype
    ):
        ##### Cosmos pipeline model
        model = Text2ImagePipeline.from_config(config, dit_path=config.dit_path, use_text_encoder=False)

        ##### Conditioning
        conditioner = model.conditioner

        def sample_batch_image(h: int = 544, w: int = 960):
            batch_size = self.batch_size
            data_batch = {
                "dataset_name": "image_data",
                "images": torch.zeros(batch_size, 3, h, w).to(device, dtype=dtype),
                "t5_text_embeddings": torch.zeros(batch_size, 512, 1024).to(device, dtype=dtype),
                "fps": torch.ones((batch_size,)).to(device, dtype=dtype) * 24,
                "padding_mask": torch.zeros(batch_size, 1, h, w).to(device, dtype=dtype),
            }
            return data_batch

        data_batch = sample_batch_image()
        is_image_batch = True

        condition, uncondition = conditioner.get_condition_uncondition(data_batch)
        del condition

        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        self.condition = uncondition

        ##### MiniTrainDIT Model
        self.unet = model

        ##### VAE Model
        self.sigma_data = model.sigma_data
        self.vae = model.tokenizer

    def initialize_cosmos_model_inference(
        self,
        config: LazyDict[Text2ImagePipelineConfig],
        vae_skip_connection: bool,
        device: str | torch.device,
        dtype: torch.dtype,
    ):
        # CUDA streams
        self.encode_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self.denoise_stream = torch.cuda.Stream()

        ##### Conditioning
        self.condition = FastUnconditioner.from_config(config, batch_size=self.batch_size, device=device, dtype=dtype)

        ##### U-Net Model
        self.unet = FastDenoiser.from_config(config, dit_path=config.dit_path).to(device=device)

        ##### VAE Model
        self.vae = FastTokenizer.from_config(config, skip_connection=vae_skip_connection).to(device=device)
        self.vae_encoder = ImageEncoder(self.vae)
        self.vae_decoder = ImageDecoder(self.vae)

    def initialize_cosmos_model(
        self, inference_only_mode: bool, vae_skip_connection: bool, device: str | torch.device, dtype: torch.dtype
    ):
        self.inference_only_mode = inference_only_mode

        if self.inference_only_mode:
            # We dont use this flag in inference only mode, just for bookkeeping
            self._vae_skip_connection = vae_skip_connection
            config = get_pipeline_config(fast_pipeline=True)
            self.initialize_cosmos_model_inference(
                config=config, vae_skip_connection=vae_skip_connection, device=device, dtype=dtype
            )
        else:
            config = get_pipeline_config(fast_pipeline=False)
            self.vae_skip_connection = vae_skip_connection
            self.initialize_cosmos_model_training(config=config, device=device, dtype=dtype)

        print("=" * 50)
        print("SUCCESS in initializing Cosmos Model")
        print(f"Number of parameters in UNet: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.2f}M")
        print(f"Number of parameters in VAE: {sum(p.numel() for p in self.vae.parameters()) / 1e6:.2f}M")
        print("=" * 50)
        self.unet.to("cuda")
        self.vae.to("cuda")

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()

        if self.train_full_unet:
            self.unet.requires_grad_(True)
        else:
            raise ValueError("!! train partial Unet not implemented")

        self.vae.train()
        self.vae.requires_grad_(True)

        for name, param in self.vae.named_parameters():
            if "time_conv" in name:
                param.requires_grad = False
                print("set ", name, "grad to be false")

        if self.freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()

        if self.freeze_vae_encoder:
            self.vae.encoder.eval()
            self.vae.encoder.requires_grad_(False)

    @torch.no_grad()
    def forward_inference_basic(self, x):
        z = self.vae_encoder(x)
        z_denoised = self.unet(z)
        o = self.vae_decoder(z_denoised)
        return o

    @torch.no_grad()
    def forward_inference(self, x):
        z_ready = torch.cuda.Event()
        with torch.cuda.stream(self.encode_stream):
            z = self.vae_encoder(x)
            z_ready.record(stream=self.encode_stream)

        z_denoised_ready = torch.cuda.Event()
        with torch.cuda.stream(self.denoise_stream):
            self.denoise_stream.wait_event(z_ready)
            z_denoised = self.unet(z)  # cannot use keyword argument here with AOTModule
            z_denoised_ready.record(stream=self.denoise_stream)

        o_ready = torch.cuda.Event()
        with torch.cuda.stream(self.decode_stream):
            self.decode_stream.wait_event(z_denoised_ready)
            o = self.vae_decoder(z_denoised)
            o_ready.record(stream=self.decode_stream)

        # Let operations on the main stream wait for the completion of the streams
        o_ready.wait()
        return o

    def forward_training(self, x, timesteps=None):
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"

        # Define the functions for the encoding and decoding steps
        def vae_encode_training(state: torch.Tensor) -> torch.Tensor:
            return self.vae.encode(state) * self.sigma_data

        def vae_decode_training(latent: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(latent / self.sigma_data)

        # Encoding
        unet_input = vae_encode_training(x[:, :, None, :, :])

        # Denoising
        sigma_B_T = self.timesteps.to(dtype=unet_input.dtype) / 1000
        z_denoised = self.unet.denoise(xt_B_C_T_H_W=unet_input, sigma=sigma_B_T, condition=self.condition).x0

        # Decoding
        if self.vae_skip_connection:
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = vae_decode_training(z_denoised)[:, :, 0]

        return output_image

    def forward(self, x, timesteps=None):
        assert len(x.shape) == 4
        if self.inference_only_mode:
            return self.forward_inference(x)
        else:
            return self.forward_training(x, timesteps=timesteps)

    def save_model(self, outf, net_disc, optimizer, optimizer_disc):
        sd = {}
        sd["state_dict_unet"] = self.unet.state_dict()
        sd["state_dict_vae"] = self.vae.state_dict()
        sd["net_disc"] = net_disc.state_dict()
        sd["optimizer"] = optimizer.state_dict()
        sd["optimizer_disc"] = optimizer_disc.state_dict()
        torch.save(sd, outf)
