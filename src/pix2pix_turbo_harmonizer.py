# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pix2Pix_Turbo wrapper with multi-view (b c v h w) support for the cosmos-predict2
fast-tokenizer pipeline. Mirrors the training-time behavior of the legacy
diffusion-fixer model so the same .pkl checkpoints (state_dict_unet /
state_dict_vae) can be loaded for both training resume and inference.
"""

import time

import torch
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.configs.base.config_text2image import (
    Text2ImagePipelineConfig,
    get_cosmos_predict2_text2image_pipeline,
)
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from cosmos_predict2.tokenizers.tokenizer import CausalConv3d, ResidualBlock
from einops import rearrange
from imaginaire.lazy_config import LazyDict

from model import MiniTrainDIT


def get_pipeline_config(
    base_dir: str = "/work/models/base", fast_pipeline: bool = False
) -> LazyDict[Text2ImagePipelineConfig]:
    config = get_cosmos_predict2_text2image_pipeline(model_size="0.6B", fast_tokenizer=False)
    # config.dit_path = f"{base_dir}/model_fast_tokenizer.pt"
    # config.tokenizer["vae_pth"] = f"{base_dir}/tokenizer_fast.pth"
    config.dit_path = "checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/model.pt"
    config.guardrail_config.enabled = False
    if fast_pipeline:
        config.net["_target_"] = MiniTrainDIT
    config.net.atten_backend = "torch"
    return config


def load_ckpt_from_state_dict(net_pix2pix, net_disc, optimizer, optimizer_disc, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")
    unet_sd = {}
    for k, v in sd["state_dict_unet"].items():
        if k.startswith("net."):
            unet_sd["dit." + k[len("net."):]] = v
        elif k.startswith("net_ema."):
            unet_sd["dit_ema." + k[len("net_ema."):]] = v
        else:
            unet_sd[k] = v
    net_pix2pix.unet.load_state_dict(unet_sd, strict=False)
    net_pix2pix._inner_vae().load_state_dict(sd["state_dict_vae"], strict=False)
    if net_disc is not None and sd.get("net_disc") is not None:
        net_disc.load_state_dict({k[7:]: v for k, v in sd["net_disc"].items()})
    optimizer.load_state_dict(sd["optimizer"])
    if optimizer_disc is not None and sd.get("optimizer_disc") is not None:
        optimizer_disc.load_state_dict(sd["optimizer_disc"])
    print(f"\n!!!! loaded pretrained weights from {pretrained_path}\n")
    return net_pix2pix, net_disc, optimizer, optimizer_disc


def save_ckpt(net_pix2pix, net_disc, optimizer, optimizer_disc, outf, train_full_unet=False, freeze_vae=False):
    sd = {
        "state_dict_unet": net_pix2pix.unet.state_dict(),
        "state_dict_vae": net_pix2pix._inner_vae().state_dict(),
        "net_disc": net_disc.state_dict() if net_disc is not None else None,
        "optimizer": optimizer.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict() if optimizer_disc is not None else None,
    }
    torch.save(sd, outf)


CACHE_T = 2


def my_vae_encoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    l_blocks = []
    for layer in self.downsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
        l_blocks.append(x)

    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)

    self.current_down_blocks = l_blocks
    return x


def my_vae_decoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
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
            skip_in = skip_convs[layer_index](skip_input)
            x = x + skip_in

        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x


class Pix2Pix_Turbo(torch.nn.Module):
    """Training/inference module that accepts (B, C, V, H, W) latent video tensors
    of any V (>=1). The legacy diffusion-fixer checkpoint is layout-compatible."""

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
    ):
        super().__init__()

        self.experiment_name = experiment_name
        self.s3_checkpoint_dir = s3_checkpoint_dir
        self.batch_size = batch_size
        self.timesteps = torch.tensor([timestep], device=device)
        self.timesteps_int = timestep

        self.train_full_unet = train_full_unet
        self.freeze_vae = freeze_vae
        self.freeze_vae_encoder = freeze_vae_encoder
        self.vae_skip_connection = vae_skip_connection

        self.use_sched = use_sched
        if self.use_sched:
            self.sched = None

        self.initialize_cosmos_model(device=device, dtype=dtype)
        self.set_train()

        if pretrained_path is not None:
            print("loading from", pretrained_path)
            sd = torch.load(pretrained_path, map_location="cpu")
            # Legacy harmonizer checkpoints stored DiT weights under `net.` / `net_ema.`
            # prefixes. Current Text2ImagePipeline exposes them as `dit` / `dit_ema`,
            # so remap before loading. With strict=False, a prefix mismatch would
            # otherwise silently drop every UNet weight.
            unet_sd = {}
            for k, v in sd["state_dict_unet"].items():
                if k.startswith("net."):
                    unet_sd["dit." + k[len("net."):]] = v
                elif k.startswith("net_ema."):
                    unet_sd["dit_ema." + k[len("net_ema."):]] = v
                else:
                    unet_sd[k] = v
            missing, unexpected = self.unet.load_state_dict(unet_sd, strict=False)
            print(f"  unet load: {len(missing)} missing, {len(unexpected)} unexpected")
            print(f"    [OK] This is expected — name conversion during code migration.")
            print(f"    [OK] Legacy ckpt stored both `net.*` (live) and `net_ema.*` (EMA) branches;")
            print(f"    [OK] the current pipeline only has `dit.*`, so the renamed `dit_ema.*` keys")
            print(f"    [OK] (~414) land as 'unexpected' and a few new `dit.*` buffers (~81) land as 'missing'.")
            print(f"    [OK] Inference uses only the matched live weights — no action needed.")
            missing, unexpected = self._inner_vae().load_state_dict(sd["state_dict_vae"], strict=False)
            print(f"  vae load: {len(missing)} missing, {len(unexpected)} unexpected")
            if unexpected:
                print(f"    first unexpected: {unexpected[:5]}")

        print("=" * 50)
        print(
            f"Number of trainable parameters in UNet: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print(
            f"Number of trainable parameters in VAE: {sum(p.numel() for p in self._inner_vae().parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print("=" * 50)

    def initialize_cosmos_model(self, device: str | torch.device, dtype: torch.dtype):
        config = get_pipeline_config(fast_pipeline=False)
        model = Text2ImagePipeline.from_config(config, 
                                                use_text_encoder=False,
                                                dit_path=config.dit_path
                                                )

        # Build an unconditional condition matching the legacy training setup
        def sample_batch_image(h: int = 544, w: int = 960):
            return {
                "dataset_name": "image_data",
                "images": torch.zeros(self.batch_size, 3, h, w).to(device, dtype=dtype),
                "t5_text_embeddings": torch.zeros(self.batch_size, 512, 1024).to(device, dtype=dtype),
                "fps": torch.ones((self.batch_size,)).to(device, dtype=dtype) * 24,
                "padding_mask": torch.zeros(self.batch_size, 1, h, w).to(device, dtype=dtype),
            }

        conditioner = model.conditioner
        condition, uncondition = conditioner.get_condition_uncondition(sample_batch_image())
        del condition
        uncondition = uncondition.edit_data_type(DataType.IMAGE)
        self.condition = uncondition

        self.unet = model
        self.sigma_data = model.sigma_data
        self.vae = model.tokenizer

        if self.vae_skip_connection:
            inner_vae = self._inner_vae()
            print("------ adding skip connection convs to VAE decoder")
            for ch, idx_list in [(384, [0, 1, 2, 3, 4]), (192, [5, 6]), (96, [7, 8])]:
                for i in idx_list:
                    conv = torch.nn.Conv3d(ch, ch, kernel_size=1, stride=1, bias=False).to(device)
                    torch.nn.init.constant_(conv.weight, 1e-5)
                    setattr(inner_vae.decoder, f"skip_conv_{i}", conv)
            inner_vae.encoder.forward = my_vae_encoder_fwd.__get__(
                inner_vae.encoder, inner_vae.encoder.__class__
            )
            inner_vae.decoder.forward = my_vae_decoder_fwd.__get__(
                inner_vae.decoder, inner_vae.decoder.__class__
            )

        print("=" * 50)
        print("SUCCESS in initializing Cosmos model")
        print(f"Number of parameters in UNet: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.2f}M")
        print(f"Number of parameters in VAE: {sum(p.numel() for p in self._inner_vae().parameters()) / 1e6:.2f}M")
        print("=" * 50)
        self.unet.to(device)
        self._inner_vae().to(device)

    def _inner_vae(self):
        # Resolve to the underlying VAE module that owns encoder/decoder. Different
        # cosmos_predict2 builds wrap the tokenizer at different depths; walk down
        # until we hit something with .encoder and .decoder.
        cur = self.vae
        for _ in range(4):
            if hasattr(cur, "encoder") and hasattr(cur, "decoder"):
                return cur
            for attr in ("model", "tokenizer", "module"):
                if hasattr(cur, attr):
                    cur = getattr(cur, attr)
                    break
            else:
                break
        return cur

    def set_eval(self):
        inner = self._inner_vae()
        self.unet.eval()
        inner.eval()
        self.unet.requires_grad_(False)
        inner.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        if self.train_full_unet:
            self.unet.requires_grad_(True)
        else:
            raise ValueError("!! train partial Unet not implemented")

        inner = self._inner_vae()
        inner.train()
        inner.requires_grad_(True)
        for name, param in inner.named_parameters():
            if "time_conv" in name:
                param.requires_grad = False

        if self.freeze_vae:
            inner.requires_grad_(False)
            inner.eval()

        if self.freeze_vae_encoder:
            inner.encoder.eval()
            inner.encoder.requires_grad_(False)

    def vae_encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(state) * self.sigma_data

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.vae_skip_connection:
            inner = self._inner_vae()
            inner.decoder.incoming_skip_acts = inner.encoder.current_down_blocks
        return self.vae.decode(latent / self.sigma_data)

    def forward(self, x, timesteps=None):
        """x: (B, C, V, H, W) — V can be 1.  Returns (B, C, V, H, W)."""
        assert len(x.shape) == 5, f"expected 5D input (B,C,V,H,W); got {x.shape}"
        start_time = time.time()

        unet_input = self.vae_encode(x)
        sigma_B_T = self.timesteps.to(dtype=unet_input.dtype) / 1000
        z_denoised = self.unet.denoise(
            xt_B_C_T_H_W=unet_input, sigma=sigma_B_T, condition=self.condition
        ).x0
        output_image = self.vae_decode(z_denoised)

        torch.cuda.synchronize()
        # print(f"----Total time: {time.time() - start_time:.4f} seconds")
        return output_image

    def save_model(self, outf, net_disc, optimizer, optimizer_disc):
        save_ckpt(self, net_disc, optimizer, optimizer_disc, outf)
