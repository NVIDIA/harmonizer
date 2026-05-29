#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ONNX Export and TensorRT Conversion Script for fixer-trt
#
# Usage:
#   ./run_export.sh                    # Default settings
#   ./run_export.sh --batch_size 4     # Custom batch size
#   ./run_export.sh --help             # Show all options

set -e

# Default settings
BATCH_SIZE=${BATCH_SIZE:-8}
HEIGHT=${HEIGHT:-544}
WIDTH=${WIDTH:-960}
TIMESTEP=${TIMESTEP:-400}
DTYPE=${DTYPE:-bfloat16}
OUTPUT_DIR=${OUTPUT_DIR:-./exported_models}
PRECISION=${PRECISION:-bf16}
SKIP_TRT=${SKIP_TRT:-0}
EXPORT_COMPONENTS=${EXPORT_COMPONENTS:-denoiser}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --timestep)
            TIMESTEP="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --skip_trt)
            SKIP_TRT=1
            shift
            ;;
        --export_components|--export-components)
            EXPORT_COMPONENTS="$2"
            shift 2
            ;;
        --help)
            echo "ONNX Export and TensorRT Conversion Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --batch_size N    Batch size for export (default: 8)"
            echo "  --height N        Input image height (default: 544)"
            echo "  --width N         Input image width (default: 960)"
            echo "  --timestep N      Diffusion timestep (default: 400)"
            echo "  --dtype TYPE      Model dtype: float32, float16, bfloat16 (default: bfloat16)"
            echo "  --output_dir DIR  Output directory (default: ./exported_models)"
            echo "  --precision P     TRT precision: fp32, fp16, bf16, int8 (default: bf16)"
            echo "  --export_components C  Components to export: all, encoder, denoiser, decoder (default: denoiser)"
            echo "  --skip_trt        Skip TensorRT conversion"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create output directories
ONNX_DIR="${OUTPUT_DIR}/onnx"
TRT_DIR="${OUTPUT_DIR}/trt"
mkdir -p "$ONNX_DIR" "$TRT_DIR"

echo "============================================================"
echo "ONNX Export and TensorRT Conversion"
echo "============================================================"
echo "Batch size:  $BATCH_SIZE"
echo "Image size:  ${WIDTH}x${HEIGHT}"
echo "Timestep:    $TIMESTEP"
echo "Dtype:       $DTYPE"
echo "TRT Precision: $PRECISION"
echo "Export components: $EXPORT_COMPONENTS"
echo "Output dir:  $OUTPUT_DIR"
echo "============================================================"

# Step 1: Export to ONNX
echo ""
echo "[Step 1/2] Exporting to ONNX..."
echo "------------------------------------------------------------"

cd "$SRC_DIR"
python -m onnx_export.export_onnx \
    --output_dir "$ONNX_DIR" \
    --batch_size "$BATCH_SIZE" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --timestep "$TIMESTEP" \
    --dtype "$DTYPE" \
    --export_components "$EXPORT_COMPONENTS" \
    --verify

if [ $? -ne 0 ]; then
    echo "ONNX export failed!"
    exit 1
fi

# Step 2: Convert to TensorRT
if [ "$SKIP_TRT" -eq 0 ]; then
    echo ""
    echo "[Step 2/2] Converting to TensorRT..."
    echo "------------------------------------------------------------"
    
    python -m onnx_export.convert_trt \
        --onnx_dir "$ONNX_DIR" \
        --output_dir "$TRT_DIR" \
        --precision "$PRECISION" \
        --components "$EXPORT_COMPONENTS"
    
    if [ $? -ne 0 ]; then
        echo "TensorRT conversion failed!"
        exit 1
    fi
else
    echo ""
    echo "[Step 2/2] Skipping TensorRT conversion (--skip_trt)"
fi

echo ""
echo "============================================================"
echo "Export completed successfully!"
echo "============================================================"
echo ""
echo "Files created:"
echo "  ONNX models: $ONNX_DIR/"
ls -la "$ONNX_DIR"/*.onnx 2>/dev/null || echo "    (no ONNX files)"
echo ""

if [ "$SKIP_TRT" -eq 0 ]; then
    echo "  TRT engines: $TRT_DIR/"
    ls -la "$TRT_DIR"/*.trt 2>/dev/null || echo "    (no TRT files)"
    echo ""
fi

echo "Next steps:"
echo "  # Run benchmark"
echo "  python -m onnx_export.inference_trt --engine_dir $TRT_DIR --benchmark"
echo ""
echo "  # Run inference"
echo "  python -m onnx_export.inference_trt --engine_dir $TRT_DIR --input image.png"
