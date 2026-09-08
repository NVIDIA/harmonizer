# Runtime Image — DiffusionHarmonizer

Harmonizer runs in a Cosmos-Predict2 environment built on top of the
NVIDIA PyTorch base image. For repeat inference, build the project
image once from the public `NVIDIA/harmonizer` checkout instead of
installing packages interactively on every run.

## Build The Image

```bash
CODE_DIR=$PWD/harmonizer
git clone https://github.com/NVIDIA/harmonizer.git "$CODE_DIR"
cd "$CODE_DIR"

docker build \
  -t harmonizer-cosmos-env \
  -f Dockerfile.cosmos \
  .
```

`Dockerfile.cosmos` uses `nvcr.io/nvidia/pytorch:25.10-py3` as its base,
installs `cosmos-predict2` plus the checkout's `requirements.txt`, and
applies `tokenizer.patch` and `text2image_dit.patch` during the build.

If your site requires authenticated NGC pulls for the base image, login
first:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Do not use `:latest` in reproducible workflows.

## Download Checkpoints

Inference needs both the Harmonizer checkpoints and the base Cosmos
model. Run the repo helper from the checkout root so the relative paths
match what the code loads:

```bash
./download_checkpoints.sh
```

This populates `$CODE_DIR/models/diffusion_harmonizer.pkl` and
`$CODE_DIR/src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/`. The
`-v "$CODE_DIR":/work` mount then exposes both at `/work/models/...` and
`/work/src/checkpoints/...`. See
[`inference.md § Checkpoint download`](inference.md#checkpoint-download).

## Run Inference

Once the image is built and checkpoints are downloaded, the canonical
`docker run` invocation and the full flag matrix for
`inference_pix2pix_turbo_harmonizer.py` live in
[`inference.md`](inference.md). The script must be launched from
`/work/src`.

## Raw Container Mode

If you run directly from `nvcr.io/nvidia/pytorch:25.10-py3` instead of
the project image, replicate the `Dockerfile.cosmos` steps: install
`cosmos-predict2` and the checkout's `requirements.txt`, then apply the
Blackwell / training patches called out in
[`inference.md § Container setup`](inference.md#container-setup). The
project Dockerfile applies these during build, so this step is only
needed in raw-container mode.

## Rebuild Policy

Rebuild the image whenever:

- The `NVIDIA/harmonizer` checkout changes.
- The base `nvcr.io/nvidia/pytorch` tag changes.
- `requirements.txt`, `tokenizer.patch`, or `text2image_dit.patch` changes.

Treat the image as disposable. It should not bake in `HF_TOKEN`,
`NGC_API_KEY`, datasets, or downloaded model checkpoints.
