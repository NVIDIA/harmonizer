#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Full ONNX export → TensorRT conversion → Benchmark pipeline
# 
# Usage:
#   ./export_and_benchmark.sh [OPTIONS]
#
# Options:
#   --batch_size N      Batch size (default: 8)
#   --height H          Image height (default: 544)
#   --width W           Image width (default: 960)
#   --timestep T        Diffusion timestep (default: 400)
#   --precision P       TensorRT precision: fp32, fp16, bf16 (default: bf16)
#   --output_dir DIR    Output directory (default: ./exported_models)
#   --skip_export       Skip ONNX export (use existing)
#   --skip_convert      Skip TensorRT conversion (use existing)
#   --benchmark_only    Only run benchmark

set -e

# Default values
BATCH_SIZE=8
HEIGHT=544
WIDTH=960
TIMESTEP=400
PRECISION="bf16"
OUTPUT_DIR="./exported_models"
SKIP_EXPORT=false
SKIP_CONVERT=false
BENCHMARK_ONLY=false

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
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_export)
            SKIP_EXPORT=true
            shift
            ;;
        --skip_convert)
            SKIP_CONVERT=true
            shift
            ;;
        --benchmark_only)
            BENCHMARK_ONLY=true
            SKIP_EXPORT=true
            SKIP_CONVERT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

ONNX_DIR="${OUTPUT_DIR}/onnx"
TRT_DIR="${OUTPUT_DIR}/trt"

echo "============================================================"
echo "fixer-trt ONNX Export & TensorRT Optimization Pipeline"
echo "============================================================"
echo "Batch size: ${BATCH_SIZE}"
echo "Image size: ${WIDTH}x${HEIGHT}"
echo "Timestep: ${TIMESTEP}"
echo "Precision: ${PRECISION}"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Step 1: Export to ONNX
if [ "$SKIP_EXPORT" = false ]; then
    echo ""
    echo "Step 1/3: Exporting to ONNX..."
    echo "------------------------------------------------------------"
    
    mkdir -p "${ONNX_DIR}"
    
    python export_onnx.py \
        --output_dir "${ONNX_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --timestep "${TIMESTEP}" \
        --dtype bfloat16 \
        --verify
    
    echo "ONNX export completed!"
else
    echo ""
    echo "Step 1/3: Skipping ONNX export (using existing)"
fi

# Step 2: Convert to TensorRT
if [ "$SKIP_CONVERT" = false ]; then
    echo ""
    echo "Step 2/3: Converting to TensorRT..."
    echo "------------------------------------------------------------"
    
    mkdir -p "${TRT_DIR}"
    
    python convert_trt.py \
        --onnx_dir "${ONNX_DIR}" \
        --output_dir "${TRT_DIR}" \
        --precision "${PRECISION}" \
        --workspace 8192 \
        --optimization_level 5
    
    echo "TensorRT conversion completed!"
else
    echo ""
    echo "Step 2/3: Skipping TensorRT conversion (using existing)"
fi

# Step 3: Benchmark
echo ""
echo "Step 3/3: Running benchmark..."
echo "------------------------------------------------------------"

# Try TRT engines first, fall back to ONNX
if [ -f "${TRT_DIR}/denoiser.trt" ]; then
    ENGINE_DIR="${TRT_DIR}"
    echo "Using TensorRT engines from ${TRT_DIR}"
else
    ENGINE_DIR="${ONNX_DIR}"
    echo "TensorRT engines not found, using ONNX models from ${ONNX_DIR}"
fi

python inference_trt.py \
    --engine_dir "${ENGINE_DIR}" \
    --benchmark \
    --warmup_iterations 10 \
    --benchmark_iterations 100

echo ""
echo "============================================================"
echo "Pipeline completed!"
echo "============================================================"
echo ""
echo "Exported files:"
echo "  ONNX models: ${ONNX_DIR}/"
echo "  TRT engines: ${TRT_DIR}/"
echo ""
echo "To run inference on an image:"
echo "  python inference_trt.py --engine_dir ${TRT_DIR} --input image.png"
echo ""
