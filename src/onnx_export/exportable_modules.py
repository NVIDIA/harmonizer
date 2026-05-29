# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ONNX-exportable wrapper modules for fixer-trt.

These modules wrap the original FastTokenizer and FastDenoiser to make them
compatible with ONNX export by:
1. Baking cached parameters as registered buffers
2. Simplifying forward passes for static graph export
3. Removing dynamic control flow
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class OnnxExportableVAEEncoder(nn.Module):
    """
    ONNX-exportable VAE Encoder wrapper.
    
    Wraps FastTokenizer.encode() with proper dimension handling for ONNX export.
    Input: [B, 3, H, W] image tensor
    Output: [B, 16, 1, H//8, W//8] latent tensor
    """
    
    def __init__(self, fast_tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = fast_tokenizer.tokenizer
        self.sigma_data = fast_tokenizer.sigma_data
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape [B, 3, H, W]
        Returns:
            Latent tensor of shape [B, 16, 1, H//8, W//8]
        """
        # Add time dimension: [B, 3, H, W] -> [B, 3, 1, H, W]
        x = x.unsqueeze(2)
        # Encode and scale
        latent = self.tokenizer.encode(x) * self.sigma_data
        return latent


class OnnxExportableVAEDecoder(nn.Module):
    """
    ONNX-exportable VAE Decoder wrapper.
    
    Wraps FastTokenizer.decode() with proper dimension handling for ONNX export.
    Note: Skip connection is NOT supported in ONNX export mode.
    
    Input: [B, 16, 1, H//8, W//8] latent tensor
    Output: [B, 3, H, W] image tensor
    """
    
    def __init__(self, fast_tokenizer: nn.Module, skip_connection: bool = False):
        super().__init__()
        self.tokenizer = fast_tokenizer.tokenizer
        self.sigma_data = fast_tokenizer.sigma_data
        
        if skip_connection:
            raise ValueError(
                "Skip connection is not supported in ONNX export mode. "
                "The encoder and decoder are separate ONNX graphs and cannot share activations. "
                "Set skip_connection=False for ONNX export."
            )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent tensor of shape [B, 16, 1, H//8, W//8]
        Returns:
            Output image tensor of shape [B, 3, H, W]
        """
        # Decode and scale
        output = self.tokenizer.decode(latent / self.sigma_data)
        # Remove time dimension: [B, 3, 1, H, W] -> [B, 3, H, W]
        output = output[:, :, 0]
        return output


class OnnxExportableDenoiser(nn.Module):
    """
    ONNX-exportable Denoiser (DiT) wrapper.
    
    This module bakes all cached parameters (scaling coefficients, timestep embeddings,
    AdaLN parameters) as registered buffers, making the model fully static and
    ONNX-exportable.
    
    Input: [B, 16, 1, H//8, W//8] noisy latent tensor
    Output: [B, 16, 1, H//8, W//8] denoised latent tensor
    """
    
    def __init__(
        self,
        fast_denoiser: nn.Module,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Args:
            fast_denoiser: The FastDenoiser module (must have compute_caching called)
            batch_size: Fixed batch size for ONNX export
            height: Input image height
            width: Input image width
            device: Target device
            dtype: Target dtype
        """
        super().__init__()
        
        # Copy the DiT model
        self.dit = fast_denoiser.dit
        
        # Register cached scaling parameters as buffers
        self.register_buffer("c_skip", fast_denoiser.c_skip_B_1_T_1_1.clone())
        self.register_buffer("c_out", fast_denoiser.c_out_B_1_T_1_1.clone())
        self.register_buffer("c_in", fast_denoiser.c_in_B_1_T_1_1.clone())
        
        # Store tensor kwargs
        self.tensor_kwargs = {"device": device, "dtype": dtype}
        
        # Store fixed parameters for ONNX
        self.register_buffer("fps", fast_denoiser.fps.clone())
        
        # padding_mask might be None
        if fast_denoiser.padding_mask is not None:
            self.register_buffer("padding_mask", fast_denoiser.padding_mask.clone())
        else:
            self.padding_mask = None
            
        self.data_type = fast_denoiser.data_type
        
        # Validate batch size matches
        cached_batch_size = fast_denoiser.c_skip_B_1_T_1_1.shape[0]
        if cached_batch_size != batch_size:
            raise ValueError(
                f"Batch size mismatch: cached parameters have batch_size={cached_batch_size}, "
                f"but requested batch_size={batch_size}. "
                f"Re-initialize the model with the correct batch size before export."
            )
        
    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass with baked parameters.
        
        Args:
            xt: Noisy latent tensor of shape [B, 16, 1, H//8, W//8]
        Returns:
            Denoised latent tensor of shape [B, 16, 1, H//8, W//8]
        """
        # Apply input scaling
        scaled_input = (xt * self.c_in).to(**self.tensor_kwargs)
        
        # Forward through DiT (parameters already cached in blocks)
        net_output = self.dit(
            x_B_C_T_H_W=scaled_input,
            fps=self.fps,
            padding_mask=self.padding_mask,
            data_type=self.data_type,
            use_cuda_graphs=False,
        ).float()
        
        # Compute final prediction: x0 = c_skip * xt + c_out * net_output
        x0_pred = self.c_skip * xt + self.c_out * net_output
        
        return x0_pred


class OnnxExportableFullModel(nn.Module):
    """
    ONNX-exportable full model combining encoder, denoiser, and decoder.
    
    This is useful for exporting a single ONNX file for the entire pipeline,
    though separate exports are recommended for flexibility.
    
    Note: Skip connection is NOT supported.
    
    Input: [B, 3, H, W] input image
    Output: [B, 3, H, W] output image
    """
    
    def __init__(
        self,
        encoder: OnnxExportableVAEEncoder,
        denoiser: OnnxExportableDenoiser,
        decoder: OnnxExportableVAEDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.denoiser = denoiser
        self.decoder = decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode -> denoise -> decode
        
        Args:
            x: Input image tensor of shape [B, 3, H, W]
        Returns:
            Output image tensor of shape [B, 3, H, W]
        """
        # Encode
        latent = self.encoder(x)
        # Denoise
        denoised_latent = self.denoiser(latent)
        # Decode
        output = self.decoder(denoised_latent)
        return output


def create_exportable_modules(
    model,  # Pix2Pix_Turbo instance
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[OnnxExportableVAEEncoder, OnnxExportableDenoiser, OnnxExportableVAEDecoder]:
    """
    Create ONNX-exportable modules from a Pix2Pix_Turbo model.
    
    The model must be in inference_only_mode with compute_caching already called.
    
    Args:
        model: Pix2Pix_Turbo instance
        batch_size: Fixed batch size for ONNX export
        height: Input image height
        width: Input image width
        device: Target device
        dtype: Target dtype
        
    Returns:
        Tuple of (encoder, denoiser, decoder) modules ready for ONNX export
    """
    if not getattr(model, 'inference_only_mode', False):
        raise ValueError(
            "Model must be in inference_only_mode for ONNX export. "
            "Initialize with inference_only_mode=True."
        )
    
    # Check if skip_connection is enabled
    skip_connection = getattr(model, '_vae_skip_connection', False)
    if skip_connection:
        print(
            "WARNING: Skip connection is enabled but not supported in ONNX export. "
            "The exported model will NOT use skip connections. "
            "Results may differ from the original model."
        )
    
    # Create exportable modules
    encoder = OnnxExportableVAEEncoder(model.vae)
    denoiser = OnnxExportableDenoiser(
        model.unet,
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    )
    decoder = OnnxExportableVAEDecoder(model.vae, skip_connection=False)
    
    return encoder, denoiser, decoder
