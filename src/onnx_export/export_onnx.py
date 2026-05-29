#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ONNX Export Script for fixer-trt.

This script exports the fixer-trt model components (VAE encoder, denoiser, VAE decoder)
to ONNX format for TensorRT optimization.

Usage:
    python export_onnx.py \
        --output_dir ./onnx_models \
        --batch_size 8 \
        --height 544 \
        --width 960 \
        --timestep 400 \
        --opset_version 17

The script will create:
    - onnx_models/vae_encoder.onnx (VAE encoder, optional, will support in the future)
    - onnx_models/denoiser.onnx (Denoiser, required)
    - onnx_models/vae_decoder.onnx (VAE decoder, optional, will support in the future)
    - onnx_models/export_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.onnx

# Add parent directory to path for imports
_SCRIPT_DIR = Path(__file__).parent.resolve()
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from onnx_export.exportable_modules import (
    OnnxExportableVAEEncoder,
    OnnxExportableVAEDecoder,
    OnnxExportableDenoiser,
    create_exportable_modules,
)


def export_encoder(
    encoder: OnnxExportableVAEEncoder,
    output_path: str,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    opset_version: int = 17,
    use_dynamo: bool = False,
) -> None:
    """Export VAE encoder to ONNX."""
    print(f"Exporting VAE encoder to {output_path}...")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, height, width, dtype=dtype, device=device)
    
    encoder.eval()
    with torch.no_grad():
        # IMPORTANT:
        # `torch.onnx.export` defaults to `dynamo=True` in newer PyTorch builds.
        # If we don't explicitly set it, it will first try `torch.export.export`,
        # which may fail for some third-party modules (e.g. non-Parameter weights).
        torch.onnx.export(
            encoder,
            dummy_input,
            output_path,
            dynamo=use_dynamo,
            opset_version=opset_version,
            input_names=["input_image"],
            output_names=["latent"],
            dynamic_axes=None,  # Fixed shape for TRT optimization
            do_constant_folding=True,
            verbose=False,
        )
    
    print(f"  Encoder exported successfully!")


def export_decoder(
    decoder: OnnxExportableVAEDecoder,
    output_path: str,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    opset_version: int = 17,
    use_dynamo: bool = False,
) -> None:
    """Export VAE decoder to ONNX."""
    print(f"Exporting VAE decoder to {output_path}...")
    
    # Create dummy input (latent shape)
    latent_h = height // 8
    latent_w = width // 8
    dummy_input = torch.randn(batch_size, 16, 1, latent_h, latent_w, dtype=dtype, device=device)
    
    decoder.eval()
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            dummy_input,
            output_path,
            dynamo=use_dynamo,
            opset_version=opset_version,
            input_names=["latent"],
            output_names=["output_image"],
            dynamic_axes=None,
            do_constant_folding=True,
            verbose=False,
        )
    
    print(f"  Decoder exported successfully!")


def export_denoiser(
    denoiser: OnnxExportableDenoiser,
    output_path: str,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    opset_version: int = 17,
    use_dynamo: bool = False,
) -> None:
    """Export denoiser (DiT) to ONNX."""
    print(f"Exporting denoiser to {output_path}...")
    
    # Create dummy input (latent shape)
    latent_h = height // 8
    latent_w = width // 8
    dummy_input = torch.randn(batch_size, 16, 1, latent_h, latent_w, dtype=dtype, device=device)
    
    denoiser.eval()
    with torch.no_grad():
        torch.onnx.export(
            denoiser,
            dummy_input,
            output_path,
            dynamo=use_dynamo,
            opset_version=opset_version,
            input_names=["noisy_latent"],
            output_names=["denoised_latent"],
            dynamic_axes=None,
            do_constant_folding=True,
            verbose=False,
        )
    
    print(f"  Denoiser exported successfully!")


def verify_onnx(onnx_path: str, dummy_input: torch.Tensor, original_output: torch.Tensor) -> bool:
    """Verify ONNX model produces correct output."""
    try:
        import onnxruntime as ort
        
        sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        
        # Run inference
        onnx_output = sess.run(None, {input_name: dummy_input.cpu().numpy()})[0]
        onnx_output_tensor = torch.from_numpy(onnx_output).to(original_output.device)
        
        # Compare
        max_diff = (onnx_output_tensor - original_output).abs().max().item()
        mean_diff = (onnx_output_tensor - original_output).abs().mean().item()
        
        print(f"  Verification: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        # Tolerance for bf16/fp16
        tolerance = 1e-2
        if max_diff < tolerance:
            print(f"  PASSED (within tolerance {tolerance})")
            return True
        else:
            print(f"  WARNING: Difference exceeds tolerance {tolerance}")
            return False
            
    except ImportError:
        print("  Skipping verification (onnxruntime not installed)")
        return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export fixer-trt model to ONNX")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./onnx_models",
                        help="Output directory for ONNX models")
    
    # Model settings
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for export (fixed)")
    parser.add_argument("--height", type=int, default=544,
                        help="Input image height")
    parser.add_argument("--width", type=int, default=960,
                        help="Input image width")
    parser.add_argument("--timestep", type=int, default=400,
                        help="Diffusion timestep")
    
    # Export settings
    parser.add_argument("--opset_version", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--use_dynamo", action="store_true",
                        help="Use TorchDynamo for export (experimental)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model precision")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for model loading")
    
    # Model paths
    parser.add_argument("--model_dir", type=str, default="/work/models/base",
                        help="Directory containing model weights")
    
    # Verification
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX models with onnxruntime")
    # NOTE: In this repo's default Cosmos tokenizer checkpoint, the VAE encoder/decoder
    # are TorchScript modules that are not reliably exportable via the legacy ONNX
    # tracer (and `torch.export` also fails due to non-Parameter weights).
    # Exporting the denoiser alone is still useful for TensorRT acceleration and
    # benchmarking, so we default to "denoiser".
    parser.add_argument("--export_components", type=str, default="denoiser",
                        choices=["all", "encoder", "decoder", "denoiser"],
                        help="Which components to export")
    
    args = parser.parse_args()
    
    # Setup
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ONNX Export for fixer-trt")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Timestep: {args.timestep}")
    print(f"Dtype: {args.dtype}")
    print(f"Device: {args.device}")
    print(f"Opset version: {args.opset_version}")
    print(f"Use dynamo: {args.use_dynamo}")
    print("=" * 60)
    
    # Import and initialize model
    print("\nLoading model...")
    from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo
    
    # Initialize model
    model = Pix2Pix_Turbo(
        batch_size=args.batch_size,
        timestep=args.timestep,
        device=device,
        dtype=dtype,
        inference_only_mode=True,
        vae_skip_connection=False,  # Must be False for ONNX export
    )
    
    # Call compute_caching to bake parameters
    print("Computing cached parameters...")
    sigma = torch.tensor([args.timestep / 1000.0] * args.batch_size, device=device, dtype=dtype)
    model.unet.compute_caching(sigma, model.condition.conditioner)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Create exportable modules
    print("\nCreating exportable modules...")
    encoder, denoiser, decoder = create_exportable_modules(
        model,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        device=device,
        dtype=dtype,
    )
    
    # Export components
    export_start = time.time()
    
    if args.export_components in ["all", "encoder"]:
        encoder_path = os.path.join(args.output_dir, "vae_encoder.onnx")
        export_encoder(
            encoder, encoder_path,
            args.batch_size, args.height, args.width,
            dtype, device, args.opset_version, args.use_dynamo
        )
        
        if args.verify:
            dummy = torch.randn(args.batch_size, 3, args.height, args.width, dtype=dtype, device=device)
            with torch.no_grad():
                original = encoder(dummy)
            verify_onnx(encoder_path, dummy, original)
    
    if args.export_components in ["all", "denoiser"]:
        denoiser_path = os.path.join(args.output_dir, "denoiser.onnx")
        export_denoiser(
            denoiser, denoiser_path,
            args.batch_size, args.height, args.width,
            dtype, device, args.opset_version, args.use_dynamo
        )
        
        if args.verify:
            latent_h, latent_w = args.height // 8, args.width // 8
            dummy = torch.randn(args.batch_size, 16, 1, latent_h, latent_w, dtype=dtype, device=device)
            with torch.no_grad():
                original = denoiser(dummy)
            verify_onnx(denoiser_path, dummy, original)
    
    if args.export_components in ["all", "decoder"]:
        decoder_path = os.path.join(args.output_dir, "vae_decoder.onnx")
        export_decoder(
            decoder, decoder_path,
            args.batch_size, args.height, args.width,
            dtype, device, args.opset_version, args.use_dynamo
        )
        
        if args.verify:
            latent_h, latent_w = args.height // 8, args.width // 8
            dummy = torch.randn(args.batch_size, 16, 1, latent_h, latent_w, dtype=dtype, device=device)
            with torch.no_grad():
                original = decoder(dummy)
            verify_onnx(decoder_path, dummy, original)
    
    export_time = time.time() - export_start
    
    # Save export configuration
    files = {}
    if args.export_components in ["all", "encoder"]:
        files["encoder"] = "vae_encoder.onnx"
    if args.export_components in ["all", "denoiser"]:
        files["denoiser"] = "denoiser.onnx"
    if args.export_components in ["all", "decoder"]:
        files["decoder"] = "vae_decoder.onnx"

    config = {
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "timestep": args.timestep,
        "dtype": args.dtype,
        "opset_version": args.opset_version,
        "export_components": args.export_components,
        "latent_channels": 16,
        "spatial_compression": 8,
        "latent_height": args.height // 8,
        "latent_width": args.width // 8,
        "vae_skip_connection": False,
        "export_time_seconds": export_time,
        "torch_version": torch.__version__,
        "files": files,
    }
    
    config_path = os.path.join(args.output_dir, "export_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Export completed!")
    print(f"Time: {export_time:.2f} seconds")
    print(f"Config saved to: {config_path}")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Convert to TensorRT:")
    print(
        f"   python convert_trt.py --onnx_dir {args.output_dir} --output_dir ./trt_engines --components {args.export_components}"
    )
    print("2. Run inference:")
    if args.export_components == "all":
        print("   python inference_trt.py --engine_dir ./trt_engines --input image.png")
    else:
        print(
            "   (only partial components were exported; use `--benchmark` to benchmark exported components, "
            "or export `--export_components all` for full pipeline inference)"
        )


if __name__ == "__main__":
    main()
