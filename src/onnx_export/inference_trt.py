#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT Inference Script for fixer-trt.

This script performs inference using TensorRT engines exported from ONNX models.

Usage:
    python inference_trt.py \
        --engine_dir ./trt_engines \
        --input image.png \
        --output output.png

    # Batch inference
    python inference_trt.py \
        --engine_dir ./trt_engines \
        --input_dir ./images \
        --output_dir ./results

    # Benchmark mode
    python inference_trt.py \
        --engine_dir ./trt_engines \
        --benchmark \
        --warmup_iterations 10 \
        --benchmark_iterations 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT/PyCUDA not available. Using ONNX Runtime fallback.")


def _trt_dtype_to_torch(dtype) -> torch.dtype:
    # NOTE: TensorRT dtype mapping (minimal set we need here).
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available")
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
        return torch.bfloat16
    raise ValueError(f"Unsupported TensorRT dtype: {dtype}")


class TensorRTTorchEngine:
    """TensorRT engine wrapper that runs on torch CUDA tensors (no host copies)."""

    def __init__(
        self,
        engine_path: str,
        device: torch.device | None = None,
        *,
        use_non_default_stream: bool = True,
    ):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")

        self.device = device or torch.device("cuda")
        if self.device.type != "cuda":
            raise ValueError(f"TensorRTTorchEngine requires CUDA device, got {self.device}")

        # Using the default CUDA stream can force extra synchronization inside
        # TensorRT's enqueueV3 path. Use a dedicated non-default stream by default.
        self._exec_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device) if use_non_default_stream else None
        )

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        self.input_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.output_name))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            raise ValueError("TensorRTTorchEngine expects a CUDA tensor input")
        if not x.is_contiguous():
            x = x.contiguous()
        if x.dtype != self.input_dtype:
            x = x.to(dtype=self.input_dtype)
        if tuple(x.shape) != tuple(self.input_shape):
            raise ValueError(f"Input shape mismatch: got {tuple(x.shape)}, expected {self.input_shape}")

        caller_stream = torch.cuda.current_stream(x.device)
        exec_stream = self._exec_stream

        if exec_stream is None:
            # Execute on the caller's current stream.
            y = torch.empty(self.output_shape, device=x.device, dtype=self.output_dtype)
            stream_handle = caller_stream.cuda_stream

            self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(y.data_ptr()))
            ok = self.context.execute_async_v3(stream_handle=stream_handle)
        else:
            # Execute on a dedicated non-default stream, with correct dependencies:
            # - exec_stream waits for caller_stream to finish producing `x`
            # - caller_stream waits for exec_stream to finish producing `y`
            exec_stream.wait_stream(caller_stream)
            with torch.cuda.stream(exec_stream):
                y = torch.empty(self.output_shape, device=x.device, dtype=self.output_dtype)
                stream_handle = exec_stream.cuda_stream

                self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
                self.context.set_tensor_address(self.output_name, int(y.data_ptr()))
                ok = self.context.execute_async_v3(stream_handle=stream_handle)
            caller_stream.wait_stream(exec_stream)

        if not ok:
            raise RuntimeError("TensorRT execution failed")
        return y


class ONNXRuntimeEngine:
    """ONNX Runtime engine wrapper as fallback."""
    
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        input_data = input_data.astype(np.float32)
        return self.session.run(None, {self.input_name: input_data})[0]


class TorchTokenizer:
    """PyTorch tokenizer wrapper (CosmosImageTokenizer)."""

    def __init__(self, tokenizer_path: str, device: torch.device, dtype: torch.dtype):
        from cosmos_predict2.tokenizers.tokenizer import CosmosImageTokenizer

        self.device = device
        self.dtype = dtype
        tok = CosmosImageTokenizer(vae_pth=tokenizer_path, dtype=dtype, squeeze_for_image=True)
        tok = tok.to(device=device, dtype=dtype)
        tok.eval()
        self.tok = tok

        # sigma_data in this repo's FastTokenizer is taken from pipeline config.
        # For Fixer, it is expected to be 1.0.
        self.sigma_data = 1.0

    @torch.no_grad()
    def encode(self, x_bchw: torch.Tensor) -> torch.Tensor:
        # [B,3,H,W] -> [B,3,1,H,W] then tokenizer.encode -> [B,16,1,h,w]
        x = x_bchw.unsqueeze(2)
        z = self.tok.encode(x) * self.sigma_data
        return z

    @torch.no_grad()
    def decode(self, z_bcthw: torch.Tensor) -> torch.Tensor:
        # tokenizer.decode -> [B,3,1,H,W] -> [B,3,H,W]
        out = self.tok.decode(z_bcthw / self.sigma_data)
        return out[:, :, 0]


class FixerTRTInference:
    """Fixer inference pipeline using TRT for denoiser and Torch for tokenizer."""
    
    def __init__(
        self,
        engine_dir: str,
        use_onnx_fallback: bool = False,
        tokenizer_path: str = "/work/models/base/tokenizer_fast.pth",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            engine_dir: Directory containing TensorRT engines or ONNX models
            use_onnx_fallback: Use ONNX Runtime instead of TensorRT
        """
        engine_dir = Path(engine_dir)
        
        # Load config
        config_path = engine_dir / "engine_config.json"
        if not config_path.exists():
            config_path = engine_dir / "export_config.json"
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.batch_size = self.config["batch_size"]
        self.height = self.config["height"]
        self.width = self.config["width"]
        self.export_components = self.config.get("export_components", "all")
        self.timestep = self.config.get("timestep", None)

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Hybrid inference requires CUDA device")
        self.dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

        # Torch tokenizer (encode/decode)
        self.tokenizer = TorchTokenizer(tokenizer_path=tokenizer_path, device=self.device, dtype=self.dtype)
        
        # Denoiser engine (TRT preferred)
        denoiser_trt = engine_dir / "denoiser.trt"
        denoiser_onnx = engine_dir / "denoiser.onnx"
        self.use_trt = False
        if (not use_onnx_fallback) and TRT_AVAILABLE and denoiser_trt.exists():
            print("Using TensorRT denoiser")
            self.denoiser = TensorRTTorchEngine(str(denoiser_trt), device=self.device)
            self.use_trt = True
        elif denoiser_onnx.exists():
            print("Using ONNX Runtime denoiser")
            self.denoiser = ONNXRuntimeEngine(str(denoiser_onnx))
        else:
            raise FileNotFoundError(f"Missing denoiser model in {engine_dir} (need denoiser.trt or denoiser.onnx)")
        
        print(f"Pipeline initialized: batch_size={self.batch_size}, size={self.width}x{self.height}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array (H, W, C) in range [0, 255]
            
        Returns:
            Preprocessed tensor (B, C, H, W) in range [-1, 1]
        """
        # Resize if needed
        if image.shape[:2] != (self.height, self.width):
            import cv2
            image = cv2.resize(image, (self.width, self.height))
        
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, 0)
        
        return image
    
    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """
        Postprocess model output.
        
        Args:
            output: Model output tensor (B, C, H, W) in range [-1, 1]
            
        Returns:
            Output image as numpy array (H, W, C) in range [0, 255]
        """
        # Remove batch dimension
        output = output[0]
        
        # CHW -> HWC
        output = np.transpose(output, (1, 2, 0))
        
        # Denormalize to [0, 255]
        output = (output * 0.5 + 0.5) * 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Run full inference pipeline.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Output image as numpy array (H, W, C)
        """
        # Preprocess (numpy -> torch)
        x_np = self.preprocess(image)  # [1,3,H,W] float32
        x = torch.from_numpy(x_np).to(device=self.device, dtype=self.dtype)

        # Encode (torch)
        z = self.tokenizer.encode(x)  # [B,16,1,h,w]

        # Denoise (TRT or ORT)
        if self.use_trt:
            z_denoised = self.denoiser(z)  # torch tensor on GPU
        else:
            z_np = z.detach().float().cpu().numpy()
            z_denoised_np = self.denoiser(z_np)
            z_denoised = torch.from_numpy(z_denoised_np).to(device=self.device, dtype=self.dtype)

        # Decode (torch)
        y = self.tokenizer.decode(z_denoised)  # [B,3,H,W]
        y_np = y.detach().float().cpu().numpy()
        return self.postprocess(y_np)
    
    def benchmark(
        self,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ) -> dict:
        """
        Benchmark the inference pipeline.
        
        Args:
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        # Create random inputs (torch for tokenizer/denoiser path)
        dummy_input = torch.randn(self.batch_size, 3, self.height, self.width, device=self.device, dtype=self.dtype)
        dummy_latent = torch.randn(
            self.batch_size, 16, 1, self.height // 8, self.width // 8, device=self.device, dtype=self.dtype
        )
        
        results = {}
        
        # Helper: time a callable under torch sync
        def _timeit(fn):
            for _ in range(warmup_iterations):
                fn()
            torch.cuda.synchronize(self.device)
            times = []
            for _ in range(benchmark_iterations):
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize(self.device)
                times.append(time.perf_counter() - start)
            return times

        # Tokenizer encode
        times = _timeit(lambda: self.tokenizer.encode(dummy_input))
        results["tokenizer_encode"] = {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
        }

        # Denoiser
        if self.use_trt:
            times = _timeit(lambda: self.denoiser(dummy_latent))
        else:
            dummy_latent_np = dummy_latent.detach().float().cpu().numpy()
            times = _timeit(lambda: self.denoiser(dummy_latent_np))
        results["denoiser"] = {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
        }

        # Tokenizer decode
        times = _timeit(lambda: self.tokenizer.decode(dummy_latent))
        results["tokenizer_decode"] = {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
        }
        
        # Full pipeline (torch tokenizer + TRT/ORT denoiser)
        def _full():
            z = self.tokenizer.encode(dummy_input)
            if self.use_trt:
                z2 = self.denoiser(z)
            else:
                z2 = torch.from_numpy(self.denoiser(z.detach().float().cpu().numpy())).to(
                    device=self.device, dtype=self.dtype
                )
            _ = self.tokenizer.decode(z2)

        times = _timeit(_full)
        results["full_pipeline"] = {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
            "throughput_samples_per_sec": float(self.batch_size / np.mean(times)),
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="TensorRT inference for fixer-trt")
    
    parser.add_argument("--engine_dir", type=str, required=True,
                        help="Directory containing TensorRT engines")
    parser.add_argument("--input", type=str, default=None,
                        help="Input image path")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory for batch processing")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (used for --input_dir, and also for --input when --output is not set)")
    parser.add_argument("--use_onnx", action="store_true",
                        help="Use ONNX Runtime instead of TensorRT")
    parser.add_argument("--tokenizer_path", type=str, default="/work/models/base/tokenizer_fast.pth",
                        help="Path to Cosmos tokenizer checkpoint (.pth) for Torch tokenizer")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g. cuda:0)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Torch dtype for tokenizer and tensorRT I/O")
    
    # Benchmark options
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode")
    parser.add_argument("--warmup_iterations", type=int, default=10,
                        help="Number of warmup iterations for benchmark")
    parser.add_argument("--benchmark_iterations", type=int, default=100,
                        help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FixerTRTInference(
        engine_dir=args.engine_dir,
        use_onnx_fallback=args.use_onnx,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        dtype=args.dtype,
    )
    
    # Benchmark mode
    if args.benchmark:
        print("\n" + "=" * 60)
        print("Running benchmark...")
        print("=" * 60)
        
        results = pipeline.benchmark(
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
        
        print("\nBenchmark Results:")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"\n{name}:")
            for metric, value in metrics.items():
                if "throughput" in metric:
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value:.3f}")
        
        print("\n" + "=" * 60)
        print(f"Full Pipeline Throughput: {results['full_pipeline']['throughput_samples_per_sec']:.2f} samples/s")
        print(f"Full Pipeline Latency: {results['full_pipeline']['mean_ms']:.2f} ms")
        print("=" * 60)
        
        return
    
    # Single image inference
    if args.input:
        import cv2
        
        print(f"Processing: {args.input}")
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not read image {args.input}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        start = time.perf_counter()
        output = pipeline(image)
        elapsed = time.perf_counter() - start
        
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Determine output path:
        # - If --output is set: use it as full file path
        # - Else if --output_dir is set: write <output_dir>/<stem>_output<suffix>
        # - Else: write alongside input as <stem>_output<suffix>
        in_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            default_name = f"{in_path.stem}_output{in_path.suffix or '.png'}"
            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = out_dir / default_name
            else:
                output_path = in_path.with_name(default_name)

        cv2.imwrite(str(output_path), output)

        print(f"Output saved to: {output_path}")
        print(f"Inference time: {elapsed * 1000:.2f} ms")
        
        return
    
    # Batch processing
    if args.input_dir:
        import cv2
        from glob import glob
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            image_paths.extend(glob(str(input_dir / ext)))
        
        print(f"Found {len(image_paths)} images")
        
        total_time = 0
        for i, image_path in enumerate(image_paths):
            print(f"Processing [{i+1}/{len(image_paths)}]: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Warning: Could not read {image_path}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            start = time.perf_counter()
            output = pipeline(image)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            output_path = output_dir / Path(image_path).name
            cv2.imwrite(str(output_path), output)
        
        avg_time = total_time / len(image_paths) if image_paths else 0
        print(f"\nProcessed {len(image_paths)} images in {total_time:.2f}s")
        print(f"Average time per image: {avg_time * 1000:.2f} ms")
        print(f"Output saved to: {output_dir}")
        
        return
    
    # No input specified - show help
    parser.print_help()


if __name__ == "__main__":
    main()
