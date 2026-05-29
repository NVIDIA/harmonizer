#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT Conversion Script for fixer-trt ONNX models.

This script converts ONNX models to TensorRT engines for optimized inference.

Usage:
    python convert_trt.py \
        --onnx_dir ./onnx_models \
        --output_dir ./trt_engines \
        --precision bf16 \
        --workspace 8192

The script will create:
    - trt_engines/vae_encoder.trt
    - trt_engines/denoiser.trt
    - trt_engines/vae_decoder.trt
    - trt_engines/engine_config.json

Requirements:
    - TensorRT Python API (tensorrt)
    - Or trtexec command-line tool
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def convert_with_trtexec(
    onnx_path: str,
    engine_path: str,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    builder_optimization_level: int = 5,
    verbose: bool = False,
) -> bool:
    """
    Convert ONNX model to TensorRT engine using trtexec.
    
    Args:
        onnx_path: Path to input ONNX model
        engine_path: Path to output TensorRT engine
        precision: Precision mode (fp32, fp16, bf16, int8)
        workspace_mb: Workspace size in MB
        builder_optimization_level: Optimization level (0-5)
        verbose: Enable verbose output
    
    Returns:
        True if conversion successful, False otherwise
    """
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
        f"--builderOptimizationLevel={builder_optimization_level}",
    ]
    
    # Add precision flags
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "bf16":
        cmd.append("--bf16")
    elif precision == "int8":
        cmd.append("--int8")
        # INT8 requires calibration data, which we don't have here
        print("WARNING: INT8 requires calibration data. Using post-training quantization.")
    # fp32 is default, no flag needed
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"  Successfully converted to {engine_path}")
            return True
        else:
            print(f"  Conversion failed!")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  Conversion timed out after 1 hour")
        return False
    except FileNotFoundError:
        print("  trtexec not found. Please install TensorRT and add trtexec to PATH.")
        return False


def convert_with_python_api(
    onnx_path: str,
    engine_path: str,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    builder_optimization_level: int = 5,
) -> bool:
    """
    Convert ONNX model to TensorRT engine using Python API.
    
    This provides more control but requires TensorRT Python bindings.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("  TensorRT Python API not available. Trying trtexec...")
        return None  # Signal to try trtexec instead
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"  Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    Parse error: {parser.get_error(i)}")
            return False
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    config.builder_optimization_level = builder_optimization_level
    
    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  Enabled FP16 precision")
        else:
            print("  WARNING: FP16 not supported on this platform")
    elif precision == "bf16":
        if hasattr(trt.BuilderFlag, "BF16") and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.BF16)
            print("  Enabled BF16 precision")
        else:
            print("  WARNING: BF16 not supported, falling back to FP16")
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  Enabled INT8 precision (no calibration, using per-tensor quantization)")
        else:
            print("  WARNING: INT8 not supported on this platform")
    
    # Build engine
    print("  Building TensorRT engine (this may take a while)...")
    build_start = time.time()
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("  Failed to build engine")
        return False
    
    build_time = time.time() - build_start
    print(f"  Engine built in {build_time:.1f} seconds")
    
    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"  Saved engine to {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX models to TensorRT engines")
    
    parser.add_argument("--onnx_dir", type=str, required=True,
                        help="Directory containing ONNX models")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for TensorRT engines (default: same as onnx_dir)")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16", "int8"],
                        help="TensorRT precision")
    parser.add_argument("--workspace", type=int, default=8192,
                        help="Workspace size in MB")
    parser.add_argument("--optimization_level", type=int, default=5,
                        choices=[0, 1, 2, 3, 4, 5],
                        help="Builder optimization level (higher = slower build, faster inference)")
    parser.add_argument("--use_trtexec", action="store_true",
                        help="Force use trtexec instead of Python API")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--components", type=str, default="all",
                        choices=["all", "encoder", "decoder", "denoiser"],
                        help="Which components to convert")
    
    args = parser.parse_args()
    
    # Setup paths
    onnx_dir = Path(args.onnx_dir)
    output_dir = Path(args.output_dir) if args.output_dir else onnx_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load export config
    config_path = onnx_dir / "export_config.json"
    if config_path.exists():
        with open(config_path) as f:
            export_config = json.load(f)
        print(f"Loaded export config: batch_size={export_config['batch_size']}, "
              f"size={export_config['width']}x{export_config['height']}")
    else:
        print("WARNING: export_config.json not found")
        export_config = {}
    
    print("=" * 60)
    print("TensorRT Conversion")
    print("=" * 60)
    print(f"ONNX directory: {onnx_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Precision: {args.precision}")
    print(f"Workspace: {args.workspace} MB")
    print(f"Optimization level: {args.optimization_level}")
    print("=" * 60)
    
    # Define components to convert
    components = {
        "encoder": ("vae_encoder.onnx", "vae_encoder.trt"),
        "denoiser": ("denoiser.onnx", "denoiser.trt"),
        "decoder": ("vae_decoder.onnx", "vae_decoder.trt"),
    }
    
    if args.components != "all":
        components = {args.components: components[args.components]}
    
    # Convert each component
    results = {}
    total_start = time.time()
    
    for name, (onnx_file, trt_file) in components.items():
        onnx_path = onnx_dir / onnx_file
        engine_path = output_dir / trt_file
        
        if not onnx_path.exists():
            print(f"\nSkipping {name}: {onnx_path} not found")
            results[name] = "skipped"
            continue
        
        print(f"\nConverting {name}...")
        convert_start = time.time()
        
        success = None
        if not args.use_trtexec:
            # Try Python API first
            success = convert_with_python_api(
                str(onnx_path),
                str(engine_path),
                precision=args.precision,
                workspace_mb=args.workspace,
                builder_optimization_level=args.optimization_level,
            )
        
        if success is None or args.use_trtexec:
            # Fall back to trtexec
            success = convert_with_trtexec(
                str(onnx_path),
                str(engine_path),
                precision=args.precision,
                workspace_mb=args.workspace,
                builder_optimization_level=args.optimization_level,
                verbose=args.verbose,
            )
        
        convert_time = time.time() - convert_start
        results[name] = {
            "success": success,
            "time_seconds": convert_time,
            "engine_path": str(engine_path) if success else None,
        }
        
        if success:
            engine_size = os.path.getsize(engine_path) / (1024 * 1024)
            results[name]["engine_size_mb"] = engine_size
            print(f"  Engine size: {engine_size:.1f} MB")
    
    total_time = time.time() - total_start
    
    # Save engine config
    engine_config = {
        **export_config,
        "precision": args.precision,
        "workspace_mb": args.workspace,
        "optimization_level": args.optimization_level,
        "conversion_results": results,
        "total_conversion_time_seconds": total_time,
        "files": {
            "encoder": "vae_encoder.trt",
            "denoiser": "denoiser.trt",
            "decoder": "vae_decoder.trt",
        }
    }
    
    engine_config_path = output_dir / "engine_config.json"
    with open(engine_config_path, "w") as f:
        json.dump(engine_config, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    
    for name, result in results.items():
        if isinstance(result, dict):
            status = "SUCCESS" if result["success"] else "FAILED"
            time_str = f"{result['time_seconds']:.1f}s"
            size_str = f"{result.get('engine_size_mb', 0):.1f}MB" if result["success"] else "N/A"
            print(f"  {name}: {status} ({time_str}, {size_str})")
        else:
            print(f"  {name}: {result}")
    
    print(f"\nTotal time: {total_time:.1f} seconds")
    print(f"Config saved to: {engine_config_path}")
    print("=" * 60)
    
    print("\nNext steps:")
    print(f"  python inference_trt.py --engine_dir {output_dir} --input image.png")


if __name__ == "__main__":
    main()
