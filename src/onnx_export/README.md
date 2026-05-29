# ONNX Export for fixer-trt

This module provides utilities for exporting the fixer-trt model to ONNX format and converting to TensorRT for optimized inference.

## Overview

The export pipeline consists of three steps:
1. **Export to ONNX**: Convert PyTorch model components to ONNX format
2. **Convert to TensorRT**: Optimize ONNX models using TensorRT
3. **Inference**: Run inference using Torch tokenizer and TensorRT/ONNXRUNTIME engines for Denoiser


## Quick Start

### Using the Shell Script (Recommended)

```bash
cd /path/to/fixer-trt/codebase/src

# Default settings (batch_size=8, 960x544, timestep=400)
./onnx_export/run_export.sh

# Custom settings
./onnx_export/run_export.sh \
    --batch_size 4 \
    --height 544 \
    --width 960 \
    --timestep 400 \
    --output_dir ./my_models

# Only export ONNX (skip TensorRT conversion)
./onnx_export/run_export.sh --skip_trt
```

### Manual Steps

#### Step 1: Export to ONNX

```bash
cd /path/to/fixer-trt/codebase/src

python -m onnx_export.export_onnx \
    --output_dir ./onnx_models \
    --batch_size 8 \
    --height 544 \
    --width 960 \
    --timestep 400 \
    --dtype bfloat16 \
    --verify
```

Output files: # by default, currently only support denoiser export
- `onnx_models/denoiser.onnx`
- `onnx_models/export_config.json`

#### Step 2: Convert to TensorRT

```bash
python -m onnx_export.convert_trt \
    --onnx_dir ./onnx_models \
    --output_dir ./trt_engines \
    --precision bf16 \
    --workspace 8192
```

Or using `trtexec` directly:

```bash
trtexec \
    --onnx=./onnx_models/denoiser.onnx \
    --saveEngine=./trt_engines/denoiser.trt \
    --bf16 \
    --workspace=8192 \
    --builderOptimizationLevel=5
```

#### Step 3: Run Inference

```bash
# Single image
python -m onnx_export.inference_trt \
    --engine_dir ./trt_engines \
    --input image.png \
    --output result.png

# Batch processing
python -m onnx_export.inference_trt \
    --engine_dir ./trt_engines \
    --input_dir ./images \
    --output_dir ./results

# Benchmark
python -m onnx_export.inference_trt \
    --engine_dir ./trt_engines \
    --benchmark \
    --warmup_iterations 10 \
    --benchmark_iterations 100
```

## Important Notes

### Skip Connection Limitation

**VAE skip connection is NOT supported in ONNX export mode.**

The skip connection requires the encoder and decoder to share intermediate activations, which is not possible when they are separate ONNX graphs. If your model uses skip connections, the exported ONNX model will produce slightly different results.

### Fixed Batch Size

ONNX models are exported with a **fixed batch size**. If you need different batch sizes, you'll need to export multiple models.

### Requirements

```
# For ONNX export
torch>=2.0
onnx
onnxruntime-gpu  # For verification

# For TensorRT conversion
tensorrt>=8.6
pycuda

# Or use trtexec (comes with TensorRT installation)
```

## File Structure

```
onnx_export/
├── __init__.py              # Module exports
├── exportable_modules.py    # ONNX-exportable wrapper classes
├── export_onnx.py           # ONNX export script
├── convert_trt.py           # TensorRT conversion script
├── inference_trt.py         # TensorRT inference script
├── run_export.sh            # Convenience shell script
└── README.md                # This file
```



