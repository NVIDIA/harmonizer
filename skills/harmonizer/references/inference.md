# DiffusionHarmonizer — Checkpoint download and inference

Detail moved out of `SKILL.md`.

## Container setup

Build instructions (and the raw-base fallback) are in
[`wrapper-image.md`](wrapper-image.md). Stop there first; the rest of
this file assumes the `harmonizer-cosmos-env` image (built from the
project `Dockerfile.cosmos`, base `nvcr.io/nvidia/pytorch:25.10-py3`)
is already available.

The project `Dockerfile.cosmos` installs `cosmos-predict2` and bakes
in both the tokenizer and Text2ImageDIT patches during build. You only
need to apply the patches by hand if you run inference directly inside
the raw base container instead of the project image:

```bash
patch /usr/local/lib/python3.12/dist-packages/cosmos_predict2/tokenizers/tokenizer.py tokenizer.patch
patch /usr/local/lib/python3.12/dist-packages/cosmos_predict2/models/text2image_dit.py text2image_dit.patch
```

## Checkpoint download

Install and authenticate the Hugging Face CLI, then run the repo's
download helper from the checkout root. It fetches the Harmonizer
checkpoints **and** the base Cosmos-Predict2 model that inference
requires, placing each in the directory the code expects:

```bash
python3 -m pip install --user "huggingface_hub[cli]"
export HF_TOKEN="hf_your_token_here"        # your real token, no angle brackets
hf auth login --token "$HF_TOKEN"
./download_checkpoints.sh
```

This downloads:

- `nvidia/Harmonizer` into `models/` — including the paper checkpoint
  `models/diffusion_harmonizer.pkl` and `models/harmonizer_nontemporal.pt`.
- `nvidia/Cosmos-Predict2-0.6B-Text2Image` (DiT `model.pt` + tokenizer)
  into `src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/`.

Add `--with-dataset` to also pull the training dataset into `data/`.

Expected checkpoint:

```bash
ls -lh models/diffusion_harmonizer.pkl
ls -d src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image
```

Never commit `models/`, `src/checkpoints/`, the downloaded checkpoints,
or `HF_TOKEN`.

## Run inference (`src/inference_pix2pix_turbo_harmonizer.py`)

The inference script lives in `src/` and imports sibling modules, so it
must be launched from inside `/work/src`. It also loads the base Cosmos
model from `src/checkpoints/`, so mount the whole checkout at `/work`
(do not mount only the frames). Output frames are written to a sibling
folder named `<input_dir>_<model_identifier>` next to the input — there
is no `--output` flag.

```bash
IMAGE=harmonizer-cosmos-env
CODE_DIR=/absolute/path/to/harmonizer

docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -w /work/src \
  --entrypoint python \
  "$IMAGE" \
  inference_pix2pix_turbo_harmonizer.py \
    --input_image /work/examples \
    --model_path /work/models/diffusion_harmonizer.pkl \
    --model_identifier "harmonizer_inference" \
    --timestep 250 \
    --resolution 1024 \
    --use_sched
```

To enhance frames that live outside the checkout, mount them under
`/work` (for example `-v "$INPUT_DIR":/work/input`) and point
`--input_image /work/input`. Note where the output lands: the script writes
to a sibling of the input path, so `/work/input_<model_identifier>` resolves
inside the `$CODE_DIR` mount — i.e. it appears in your **checkout**, not
next to the original `$INPUT_DIR`. Mount a second writable directory if you
want the results beside the source frames.

The README's interactive equivalent is:

```bash
docker run --gpus=all -it --ipc=host -v "$(pwd)":/work harmonizer-cosmos-env
# then, inside the container:
cd /work/src
python inference_pix2pix_turbo_harmonizer.py \
    --input_image /work/examples \
    --model_path /work/models/diffusion_harmonizer.pkl \
    --model_identifier "harmonizer_inference" \
    --timestep 250 --resolution 1024 --use_sched
```

Flags exposed by `inference_pix2pix_turbo_harmonizer.py`:

| Flag | Default | Purpose |
|------|---------|---------|
| `--input_image` | required | Directory of input frames (`.png`, `.jpg`, `.jpeg`). |
| `--model_path` | required | Path to `diffusion_harmonizer.pkl`. |
| `--model_identifier` | `fixer` | Suffix for the output folder `<input_dir>_<model_identifier>`. README uses `harmonizer_inference`. |
| `--timestep` | `250` | Diffusion timestep used by the distilled model. |
| `--resolution` | `1024` | Internal size key. `1024` -> `1024x576`. Supported keys: `1024`, `960`, `1360`. |
| `--use_sched` | off | Present in the README command, but currently inert: `src/pix2pix_turbo_harmonizer.py:214-216` only assigns `self.sched = None` and nothing reads it. Harmless to pass; do not expect a behaviour change. |
| `--vae_skip_connection` | off | Enable the VAE skip connection. |
| `--save_video` | off | Also write an MP4 assembled from the output folder. Only `.png` and `.jpg` frames are collected (`save_folder2video`, `src/inference_pix2pix_turbo_harmonizer.py:32-41`), so `.jpeg` inputs are silently omitted from the video — use `.jpg` or `.png` if you need it. |
| `--max_frames` | very large | Maximum number of frames to process. |
| `--start_frame` | `0` | Index of the first frame to process. |
| `--skip_frames` | `1` | Process every Nth frame. |
| `--offset_list` | `-1 -2 -3 -4` | Negative offsets into previously-predicted outputs used as temporal references. |
| `--nontemporal` | off | Disable temporal conditioning; run frame-by-frame. |

The model card reports H100 testing with an inference time of about
`212 ms` per frame on one H100. Trust the measurement from your target
hardware for capacity planning.

## Temporal vs. non-temporal

`inference_pix2pix_turbo_harmonizer.py` is temporal by default: once
enough history exists it uses the four previous enhanced frames
(`--offset_list -1 -2 -3 -4`) as references, and early frames fall back
to a non-temporal pass. Because each output is fed back in as a future
reference, keep each run's output folder separate so stale temporal
outputs cannot be reused accidentally.

For unordered images, or when you want each frame enhanced
independently, add `--nontemporal` to disable temporal conditioning
entirely.
