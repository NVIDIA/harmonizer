# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

import torch
import torch_tensorrt

logger = logging.getLogger(__name__)


def load_tokenizer_jit(encoder_path, decoder_path, device):
    encoder = torch.jit.load(encoder_path, map_location=device)
    decoder = torch.jit.load(decoder_path, map_location=device)
    return encoder, decoder


def load_unet(unet_pth, device: torch.device):
    class AOTModule(torch.nn.Module):
        def __init__(self, filename: str, device: torch.device):
            super().__init__()
            self.mod = torch._inductor.aoti_load_package(filename, device_index=device.index)

        def forward(self, *args, **kwargs):
            return self.mod(*args, **kwargs)

    compiled_unet = AOTModule(unet_pth, device)
    return compiled_unet


def load_unet_pkg(unet_pth, device: torch.device):
    unet = torch.load(unet_pth, weights_only=False, map_location=device)
    return unet


class FullModel(torch.nn.Module):
    def __init__(self, encoder, decoder, unet):
        super().__init__()
        self.vae_encoder = encoder
        self.vae_decoder = decoder
        self.unet = unet

    def forward(self, x):
        return self._forward_jit(x)

    def _forward_jit(self, x):
        assert len(x.shape) == 4
        # VAE encode
        z = self.vae_encoder(x)
        # Denoise
        z_denoised = self.unet(z)
        # VAE decode, vae_skip_connection = False
        o = self.vae_decoder(z_denoised)
        return o


def load_exported_model(
    export_path: str,
    timestep: int,
    vae_skip_connection: bool,
    batch_size: int,
    image_size: tuple[int, int],  # w, h
    device: torch.device,
    dtype: torch.dtype,
) -> FullModel:
    print(f"Loading exported model from {export_path}...")

    with open(os.path.join(export_path, "meta.json")) as f:
        meta = json.load(f)

    msg = "[Fixer]: "

    def check_consistency(key, value):
        assert meta[key] == value, f"{msg} {key} mismatch: expected {value}, got {meta[key]}"

    # check if device is a cuda device
    assert device.type == "cuda", f"{msg} Expected a cuda device, got {device.type}"
    assert device.index == torch.cuda.current_device(), (
        f"{msg} Default Torch CUDA device does not match the target device."
        f" Current device is 'cuda:{torch.cuda.current_device()}', target device is '{device}'."
        f" Please set `torch.cuda.set_device('cuda:{device.index}')` before loading the model."
    )

    # NOTE: The version check is commented out here because sometimes the minor/patch version differs
    #       while the major version is the same, and it can still load and perform inference successfully.
    #       For example: 2.7.0a0+79aa17489c.nv25.4 vs. 2.7.0
    if meta["trt_version"] != torch_tensorrt.__version__:
        logger.warning(
            f"{msg} TRT version mismatch: expected {torch_tensorrt.__version__}, got {meta['trt_version']}. Significant version mismatch may lead to model loading failure."
        )
    if meta["torch_version"] != torch.__version__:
        logger.warning(
            f"{msg} Torch version mismatch: expected {torch.__version__}, got {meta['torch_version']}. Significant version mismatch may lead to model loading failure."
        )
    # check_consistency("trt_version", torch_tensorrt.__version__)
    # check_consistency("torch_version", torch.__version__)

    # compare trt major version
    assert torch_tensorrt.__version__.split(".")[0] == torch_tensorrt.__version__.split(".")[0], (
        f"{msg} TRT major version mismatch: expected {torch_tensorrt.__version__.split('.')[0]}, got {meta['trt_version'].split('.')[0]}"
    )

    check_consistency("dtype", str(dtype))
    check_consistency("batch_size", batch_size)
    check_consistency("timestep", timestep)
    check_consistency("vae_skip_connection", vae_skip_connection)
    check_consistency("image_size", list(image_size))

    encoder, decoder = load_tokenizer_jit(meta["encoder_path"], meta["decoder_path"], device)
    if meta["unet_path"].endswith(".pkg.pt"):
        unet = load_unet_pkg(meta["unet_path"], device)
    else:
        unet = load_unet(meta["unet_path"], device)
    return FullModel(encoder, decoder, unet)
