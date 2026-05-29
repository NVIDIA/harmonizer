# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Model speed on H100, half precision
# batch 1 - 38ms
# batch 7 - 24ms
# batch 8 - 24ms
# batch 64 - 23ms

import argparse
import json
import os
import sys
import warnings
from glob import glob
from typing import Literal

import imageio
import numpy as np
import torch
import torch_tensorrt
from natsort import natsorted

# Detect torch_tensorrt version for compatibility
try:
    _TRT_VERSION = tuple(int(x) for x in torch_tensorrt.__version__.split(".")[:2])
except (AttributeError, ValueError):
    _TRT_VERSION = (999, 0)
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo

# Suppress warnings
warnings.filterwarnings("ignore")
import logging

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)

# Example: Disable if output is not a TTY (e.g., redirected to a file)
# This checks if the standard output stream is connected to a TTY device.
# If not, it's likely being redirected.
disable_tqdm = not sys.stdout.isatty()


def save_folder2video(save_dir, remove_key=0):
    if remove_key == 0:
        video_file = os.path.join(save_dir, save_dir.split("/")[-1] + "_video.mp4")
    else:
        video_file = os.path.join(save_dir, "video_removekey_" + str(remove_key) + ".mp4")
    im_files = natsorted(glob(os.path.join(save_dir, "*.png")) + glob(os.path.join(save_dir, "*.jpg")))
    # Loads all images at once - can run out of memory if too many frames
    ims: list[np.ndarray] = [imageio.v2.imread(f) for f in im_files]

    # chop image if dimension not divisible by 2 to fix the ffmpeg error
    for i in range(len(ims)):
        if ims[i].shape[0] % 2 == 1:
            ims[i] = ims[i][:-1, :, :]
        if ims[i].shape[1] % 2 == 1:
            ims[i] = ims[i][:, :-1, :]

    # Results in an Array is not the same as ArrayLike annotation error, so we suppress it
    if remove_key != 0:
        ims = [ims[i] for i in range(len(ims)) if i % remove_key != 0]

    imageio.v2.mimwrite(video_file, ims, fps=30, macro_block_size=1)  # type: ignore
    imageio.v2.mimwrite(video_file.replace("video.mp4", "video_10fps.mp4"), ims[::3], fps=10, macro_block_size=1)  # type: ignore
    imageio.v2.mimwrite(video_file.replace("video.mp4", "video_15fps.mp4"), ims[::2], fps=15, macro_block_size=1)  # type: ignore


def sample_input_image(batch_size: int, h: int, w: int, dtype: torch.dtype, device: torch.device):
    return torch.randn(batch_size, 3, h, w, dtype=dtype, device=device)


def model_inference(
    model, batch_size: int, h: int, w: int, dtype: torch.dtype, device: torch.device, x: torch.Tensor | None = None
):
    output = model(x if isinstance(x, torch.Tensor) else sample_input_image(batch_size, h, w, dtype, device))
    return output


def warmup_model(
    model: torch.nn.Module,
    batch_size: int,
    h: int,
    w: int,
    dtype: torch.dtype,
    device: torch.device,
    n: int = 10,
) -> None:
    """Warmup the model with dummy inference runs.

    Args:
        model: The compiled model to warmup
        batch_size: Batch size for warmup
        h: Height dimension
        w: Width dimension
        dtype: Data type
        device: Device to run on
        n: Number of warmup iterations (default: 10)
    """
    print(f"Warming up model with {n} iterations...")
    for i in tqdm(range(n), desc="Warmup", leave=False, disable=disable_tqdm):
        model_inference(model, batch_size, h, w, dtype, device)


def speed_measure(
    model_path: str,
    timestep: int,
    vae_skip_connection: bool,
    batch_size: int,
    h: int,
    w: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int = 50,
    test_iters: int = 50,
) -> float:
    """Measure inference speed by loading model and running benchmarks.

    Args:
        model_path: Path to model checkpoint
        timestep: Diffusion timestep
        vae_skip_connection: Whether to use VAE skip connections
        batch_size: Batch size for testing
        h: Height dimension
        w: Width dimension
        dtype: Data type
        device: Device to run on
        warmup_iters: Number of warmup iterations
        test_iters: Number of test iterations

    Returns:
        Average latency per sample in seconds
    """
    print("\n" + "=" * 70)
    print("⚡ SPEED MEASUREMENT")
    print("=" * 70)

    # Load and compile model
    print("Loading model for speed test...")
    model = load_and_compile_model(
        model_path=model_path,
        timestep=timestep,
        vae_skip_connection=vae_skip_connection,
        batch_size=batch_size,
        image_size=(w, h),
        device=device,
        dtype=dtype,
        compile=True,
    )

    # Warmup
    warmup_model(model, batch_size, h, w, dtype, device, n=warmup_iters)

    # Speed test
    print(f"Running speed test with {test_iters} iterations...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    ### Create CUDA events for timing
    s_event = torch.cuda.Event(enable_timing=True)
    e_event = torch.cuda.Event(enable_timing=True)
    futures = []

    ### Benchmark the model
    s_event.record()
    for i in tqdm(range(test_iters), desc="Speed test", leave=False, disable=disable_tqdm):
        model_inference(model, batch_size, h, w, dtype, device)
    e_event.record()

    ### Collect the results
    torch.cuda.synchronize()
    cuda_time = s_event.elapsed_time(e_event) / 1000.0
    latency = cuda_time / test_iters / batch_size
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    print()
    print("=" * 70)
    print("🚀 SPEED TEST RESULTS")
    print("=" * 70)
    print(f"  Batch Size:        {batch_size}")
    print(f"  Warmup Iterations: {warmup_iters}")
    print(f"  Test Iterations:   {test_iters}")
    print(f"  Latency:           {latency:.4f} s/sample")
    print(f"  Throughput:        {1 / latency:.2f} samples/s")
    print(f"  Peak memory:       {peak_memory:.2f} MB")
    print("=" * 70)
    print()

    return latency


def get_resolution_size(resolution: int) -> tuple[int, int]:
    """Map resolution to (width, height) size tuple."""
    resolution_map = {
        960: (960, 544),
        1360: (1360, 768),
        704: (704, 384),
        512: (512, 288),
        256: (256, 144),
        1024: (1024, 576),
        1920: (1920, 1072),
    }
    assert resolution in resolution_map, (
        f"Resolution {resolution} not supported. Choose from {list(resolution_map.keys())}"
    )
    return resolution_map[resolution]


def preprocess_image(img: Image.Image, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Convert PIL image to normalized tensor."""
    c_t = transforms.ToTensor()(img)
    c_t = transforms.Normalize([0.5], [0.5])(c_t).unsqueeze(0)
    return c_t.to(device=device, dtype=dtype)


def postprocess_output(output_tensor: torch.Tensor, target_size: tuple[int, int]) -> Image.Image:
    """Convert model output tensor to PIL image."""
    output_image = output_tensor.float()
    output_image = output_image[0].cpu() * 0.5 + 0.5
    output_image = torch.clamp(output_image, 0.0, 1.0)
    output_pil = transforms.ToPILImage()(output_image)
    output_pil = output_pil.resize(target_size, Image.BILINEAR)
    return output_pil


def export_encoder_with_jit(
    model,
    batch_size: int,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    encoder_path: str,
):
    class VAE_Encoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, x):
            return self.vae.encode(x)

    with torch.no_grad():
        # Create example input
        example_input = torch.randn(batch_size, 3, image_size[1], image_size[0], dtype=dtype, device=device)

        # Create encoder module
        encoder = VAE_Encoder(model.vae)

        # JIT
        traced_encoder = torch.jit.trace(encoder, example_input, strict=False)

        traced_encoder.save(encoder_path)
        print(f"Saved encoder to {encoder_path}")


def export_decoder_with_jit(
    model,
    batch_size: int,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    decoder_path: str,
) -> None:
    class VAE_Decoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, x):
            return self.vae.decode(x)[:, :, 0]

    with torch.no_grad():
        # Create example input
        latent_shape = model.vae.get_state_shape(image_size[1], image_size[0])
        example_input = torch.randn(batch_size, *latent_shape, dtype=dtype, device=device)

        # Create decoder module
        decoder = VAE_Decoder(model.vae)

        # JIT
        traced_decoder = torch.jit.trace(decoder, example_input, strict=False)

        traced_decoder.save(decoder_path)
        print(f"Saved decoder to {decoder_path}")


def compile_model_with_dynamo(
    model: torch.nn.Module,
    batch_size: int,
    image_size: tuple[int, int],  # w, h
    device: torch.device,
    dtype: torch.dtype,
    export_path: str | None = None,
) -> torch.nn.Module:
    print(f"Compiling model with TensorRT dynamo..., dtype: {dtype}")

    settings = {
        "use_python_runtime": False,
        "enabled_precisions": {dtype},
        "immutable_weights": True,
    }

    latent_shape = model.vae.get_state_shape(image_size[1], image_size[0])
    unet_inputs = [
        torch.randn([batch_size, *latent_shape], dtype=dtype, device=device),
    ]

    # Compile the model
    unet_exported = torch.export.export(model.unet, tuple(unet_inputs))
    unet_compiled = torch_tensorrt.dynamo.compile(unet_exported, tuple(unet_inputs), **settings)

    # TODO(qi): We currently cannot compile the VAE with dynamo because it is a JIT scripted model. We need
    #           to export the VAE properly with dynamo first.
    vae_compiled = torch.compile(model.vae, backend="torch_tensorrt", mode="max-autotune", dynamic=False)

    # Export the model
    if export_path is not None:
        encoder_path = os.path.join(export_path, f"vae_encoder_{batch_size}.jit.pt")
        decoder_path = os.path.join(export_path, f"vae_decoder_{batch_size}.jit.pt")
        unet_path = os.path.join(export_path, f"exported_unet_{batch_size}_{dtype}.pt2")
        # Export the encoder and decoder
        export_encoder_with_jit(model, batch_size, image_size, device, dtype, encoder_path)
        export_decoder_with_jit(model, batch_size, image_size, device, dtype, decoder_path)
        if _TRT_VERSION >= (2, 9):
            torch_tensorrt.save(
                unet_compiled, unet_path, inputs=unet_inputs, output_format="aot_inductor", retrace=True
            )
        else:
            # torch_tensorrt < 2.9.0: doesn't support aot_inductor format
            unet_path = unet_path.replace(".pt2", ".pkg.pt")
            torch.save(unet_compiled, unet_path)
        print(f"Saved denoiser to {unet_path}")
        # Also record the meta data
        meta = {
            "encoder_path": encoder_path,
            "decoder_path": decoder_path,
            "unet_path": unet_path,
            "timestep": model.timesteps_int,
            "vae_skip_connection": model._vae_skip_connection,
            "batch_size": batch_size,
            "image_size": image_size,
            "device": str(device),
            "dtype": str(dtype),
            "trt_version": torch_tensorrt.__version__,
            "torch_version": torch.__version__,
        }
        with open(os.path.join(export_path, "meta.json"), "w") as f:
            json.dump(meta, f)

    # Put together the compiled model
    model.unet = unet_compiled
    model.vae = vae_compiled
    return model


def compile_model_and_maybe_export(
    model: torch.nn.Module,
    batch_size: int,
    image_size: tuple[int, int],  # w, h
    device: torch.device,
    dtype: torch.dtype,
    compile_mode: Literal["dynamo", "inductor"] = "dynamo",
    export_path: str | None = None,
) -> torch.nn.Module:
    # Compile the model with dynamo
    if compile_mode == "dynamo":
        model = compile_model_with_dynamo(model, batch_size, image_size, device, dtype, export_path)

    # Compile the model with inductor
    elif compile_mode == "inductor":
        model = torch.compile(model, backend="inductor")
        if export_path is not None:
            raise ValueError("Exporting model with inductor compiler is not supported")

    # Raise an error for invalid compile mode
    else:
        raise ValueError(f"Invalid compile mode: {compile_mode}")

    return model


def load_and_compile_model(
    model_path: str,
    timestep: int,
    vae_skip_connection: bool,
    batch_size: int,
    image_size: tuple[int, int],  # w, h
    device: torch.device,
    dtype: torch.dtype,
    compile: bool = True,
    compile_mode: Literal["dynamo", "inductor"] = "dynamo",
) -> torch.nn.Module:
    """Initialize and export the model.

    Args:
        model_path: Path to model checkpoint
        timestep: Diffusion timestep
        vae_skip_connection: Whether to use VAE skip connections
        batch_size: Batch size for inference
        image_size: Image size (width, height)
        device: Device to run on
        dtype: Data type
        compile: Whether to compile the model for faster inference
        compile_mode: Compilation mode
    """

    model = Pix2Pix_Turbo(
        pretrained_path=model_path,
        timestep=timestep,
        vae_skip_connection=vae_skip_connection,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        inference_only_mode=True,
    )
    model.set_eval()

    if compile:
        model = compile_model_and_maybe_export(model, batch_size, image_size, device, dtype, compile_mode)

    return model


def process_single_image(
    model: torch.nn.Module,
    image_path: str,
    resolution: int,
    batch_size: int,
    h: int,
    w: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Image.Image, str]:
    """Process a single image through the model.

    Returns:
        tuple: (output_pil_image, basename)
    """
    size = get_resolution_size(resolution)

    input_image = Image.open(image_path).convert("RGB")
    original_shape = input_image.size  # w, h
    input_image = input_image.resize(size, Image.BILINEAR)

    bname = os.path.basename(image_path)

    with torch.no_grad():
        c_t = preprocess_image(input_image, device, dtype)
        output_tensor = model_inference(model, batch_size, h, w, dtype, device, x=c_t)
        output_pil = postprocess_output(output_tensor, original_shape)

    return output_pil, bname


def get_image_paths(input_dir: str, max_frames: int = None, skip_frames: int = 1) -> list[str]:
    """Get sorted list of image paths from directory."""
    all_img_paths = glob(input_dir + "/*.png") + glob(input_dir + "/*.jpg") + glob(input_dir + "/*.jpeg")
    all_img_paths.sort()

    if max_frames is None:
        max_frames = len(all_img_paths)

    return all_img_paths[:max_frames][::skip_frames]


def inference(
    model_path: str,
    timestep: int,
    vae_skip_connection: bool,
    input_dir: str,
    output_dir: str,
    resolution: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    max_frames: int = None,
    skip_frames: int = 1,
    save_video: bool = False,
    warmup_iters: int = 10,
):
    """Run inference on a directory of images.

    Args:
        model_path: Path to model checkpoint
        timestep: Diffusion timestep
        vae_skip_connection: Whether to use VAE skip connections
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        resolution: Target resolution
        batch_size: Batch size for inference
        h: Height dimension
        w: Width dimension
        dtype: Data type
        device: Device to run on
        max_frames: Maximum number of frames to process
        skip_frames: Frame skip interval
        save_video: Whether to save output as video
        warmup_iters: Number of warmup iterations
    """
    print("\n" + "=" * 70)
    print("🎨 INFERENCE MODE")
    print("=" * 70)

    image_size = get_resolution_size(resolution)
    w, h = image_size

    # Load and compile model
    print("Loading model for inference...")
    model = load_and_compile_model(
        model_path=model_path,
        timestep=timestep,
        vae_skip_connection=vae_skip_connection,
        batch_size=batch_size,
        image_size=image_size,
        device=device,
        dtype=dtype,
        compile=True,
    )

    # Warmup the model
    warmup_model(model, batch_size, h, w, dtype, device, n=warmup_iters)

    # Get image paths
    image_paths = get_image_paths(input_dir, max_frames=max_frames, skip_frames=skip_frames)

    print(f"\nProcessing {len(image_paths)} images...")
    print(f"Batch size: {batch_size}, Resolution: {resolution}\n")

    # Process images
    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Processing images"):
        output_pil, bname = process_single_image(model, img_path, resolution, batch_size, h, w, dtype, device)
        sv_path = os.path.join(output_dir, bname)
        output_pil.save(sv_path)

    print(f"\n✓ Processed {len(image_paths)} images -> {output_dir}")

    if save_video:
        save_folder2video(output_dir)
        print(f"✓ Video saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on an image.")

    # Model arguments
    parser.add_argument("--model", type=str, default=None, help="path to a model state dict to be used")
    parser.add_argument("--timestep", type=int, default=400, help="Diffusion timestep")
    parser.add_argument("--vae_skip_connection", "--vae-skip-connection", action="store_true")

    # Inference arguments
    parser.add_argument("--resolution", type=int, default=1024, help="The resolution of the output images")
    parser.add_argument("--input", type=str, default=None, help="The directory containing the input images")
    parser.add_argument("--output", type=str, default="output", help="The directory to save the output images")
    parser.add_argument(
        "--save_video", "--save-video", action="store_true", help="Whether to save the output as a video"
    )
    parser.add_argument(
        "--max_frames", "--max-frames", type=int, default=3000000, help="The maximum number of frames to process"
    )
    parser.add_argument(
        "--skip_frames", "--skip-frames", type=int, default=1, help="The interval between frames to skip"
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8, help="The batch size for inference")

    # Speed test arguments
    parser.add_argument("--test-speed", action="store_true", help="Run speed benchmark before inference")
    parser.add_argument(
        "--no-generate-images", action="store_true", help="Skip image generation (just perform speed test)"
    )
    parser.add_argument("--speed-test-iters", type=int, default=50, help="Number of iterations for speed test")
    parser.add_argument("--warmup-iters", type=int, default=50, help="Number of warmup iterations")

    # Export arguments
    parser.add_argument("--export-path", type=str, default=None, help="Path to export the compiled model")

    # Parse arguments
    args = parser.parse_args()

    # Set up device and data type
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    image_size = get_resolution_size(args.resolution)

    # Optional speed test (uses args.batch_size)
    if args.test_speed:
        speed_measure(
            model_path=args.model,
            timestep=args.timestep,
            vae_skip_connection=args.vae_skip_connection,
            batch_size=args.batch_size,
            h=image_size[1],
            w=image_size[0],
            dtype=dtype,
            device=device,
            warmup_iters=args.warmup_iters,
            test_iters=args.speed_test_iters,
        )

    # Run inference (uses args.batch_size)
    if not args.no_generate_images:
        assert args.input is not None, "Input directory is required for image generation"
        inference(
            model_path=args.model,
            timestep=args.timestep,
            vae_skip_connection=args.vae_skip_connection,
            input_dir=args.input,
            output_dir=args.output,
            resolution=args.resolution,
            batch_size=1,
            dtype=dtype,
            device=device,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames,
            save_video=args.save_video,
            warmup_iters=args.warmup_iters,
        )

    # Export the model
    if args.export_path is not None:
        os.makedirs(args.export_path, exist_ok=True)
        model = Pix2Pix_Turbo(
            pretrained_path=args.model,
            timestep=args.timestep,
            vae_skip_connection=args.vae_skip_connection,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
            inference_only_mode=True,
        )
        model.set_eval()
        compile_model_and_maybe_export(
            model,
            args.batch_size,
            image_size,
            device,
            dtype,
            export_path=args.export_path,
        )


if __name__ == "__main__":
    main()
