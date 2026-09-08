# Teardown — harmonizer / DiffusionHarmonizer

A complete DiffusionHarmonizer workflow can leave **100 GB+** on disk,
especially if you build the runtime image and download the optional
training dataset.

| Artifact | Approximate size | Source |
|----------|------------------|--------|
| `harmonizer-cosmos-env` image (base `nvcr.io/nvidia/pytorch:25.10-py3`) | tens of GB | `docker build` / `docker pull` |
| Docker build cache | tens of GB | `docker build -f Dockerfile.cosmos` |
| `harmonizer/` code checkout | repo-dependent | `git clone https://github.com/NVIDIA/harmonizer.git` |
| `models/` with `diffusion_harmonizer.pkl` | model-dependent | `./download_checkpoints.sh` (`nvidia/Harmonizer`) |
| `src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/` | model-dependent | `./download_checkpoints.sh` (base Cosmos model) |
| Hugging Face hub cache | model/dataset-dependent | `hf download` |
| `data/` training dataset | large | `./download_checkpoints.sh --with-dataset` (`nvidia/Harmonizer-Dataset`) |
| Enhanced/evaluation outputs | sequence-dependent | inference/evaluation runs |

## Reclaim Disk

Run only the blocks that apply to your host.

### 1. Container Images And Build Cache

Only the first line is specific to this skill. The rest reach beyond it —
`nvcr.io/nvidia/pytorch:25.10-py3` is a shared NVIDIA base image other
projects commonly use, and both `prune` commands operate **globally** on the
Docker daemon, removing dangling images and build cache belonging to anything
else on the host.

```bash
# Always safe: the image this skill builds
docker image rm harmonizer-cosmos-env 2>/dev/null || true

# OPT-IN — shared base image. Skip if any other project uses it:
#   docker ps -a --filter ancestor=nvcr.io/nvidia/pytorch:25.10-py3
# docker image rm nvcr.io/nvidia/pytorch:25.10-py3

# OPT-IN — these are GLOBAL and affect unrelated projects. Review first with
# `docker image ls -f dangling=true` and `docker builder du`:
# docker image prune -f
# docker builder prune -f
```

### 2. Code Checkout, Model, And Dataset Copies

Resolve and verify the target before deleting anything — these are
recursive removals of a path you supply.

Note what is **not** inside the checkout: the credential file created during
setup (`~/.harmonizer.env` by default) deliberately lives outside it, so
nothing here removes it implicitly. It is handled as its own step below, and
it holds live tokens until you do.

```bash
HARMONIZER_CHECKOUT=$(cd /absolute/path/to/harmonizer && pwd)
test -f "$HARMONIZER_CHECKOUT/download_checkpoints.sh" || {
    echo "not a harmonizer checkout: $HARMONIZER_CHECKOUT"; return 2>/dev/null || exit 1; }
cd "$HARMONIZER_CHECKOUT/.."

rm -rf "$HARMONIZER_CHECKOUT/models"
rm -rf "$HARMONIZER_CHECKOUT/src/checkpoints"
rm -rf "$HARMONIZER_CHECKOUT/data"

# The checkout itself, last
rm -rf "$HARMONIZER_CHECKOUT"

# The credential file. This lives OUTSIDE the checkout by design (see
# scripts/.env.example), so removing the checkout does NOT remove it and it
# would otherwise be left behind holding live tokens.
rm -f ~/.harmonizer.env          # adjust if you put it elsewhere
```

Do not revoke `HF_TOKEN` or `NGC_API_KEY` as part of routine teardown —
they are per-user and shared across other work. Rotate them only if you
believe a value was exposed.

If you downloaded model or dataset artifacts elsewhere, remove those
paths instead.

### 3. Hugging Face Cache

For targeted cleanup, inspect the cache first:

```bash
hf cache ls | grep -E 'Harmonizer|Cosmos-Predict2' || true
```

Then delete the specific cached repos with `hf cache rm` (optionally
`--cache-dir` to target a non-default cache), or remove only the known repo
cache directories if you are sure they are not shared by another workflow.
Resolve the effective cache root the same way the tools do, otherwise you
risk clearing the wrong one:

```bash
HF_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}}"
HF_CACHE="${HF_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/hub}"
```

### 4. Outputs

```bash
rm -rf /absolute/path/to/enhanced_frames
rm -rf /absolute/path/to/test_dataset/evaluation
rm -f /absolute/path/to/test_dataset/metrics.yaml
```

## Outputs Already Owned By Root

If a previous `docker run` omitted `-u $(id -u):$(id -g)`, fix ownership
before deleting or editing outputs:

```bash
# Take ownership of ONLY root-owned entries, on this filesystem, without
# following symlinks out of the tree. A recursive chown would also seize
# files belonging to collaborators.
TARGET=/absolute/path/to/enhanced_frames
sudo find "$TARGET" -xdev -user root -exec chown -h -- "$(id -u):$(id -g)" {} +
rm -rf /absolute/path/to/enhanced_frames
```

## Verify

```bash
docker images | grep -E 'harmonizer-cosmos-env|nvcr.io/nvidia/pytorch' || echo "images: clean"
du -sh /absolute/path/to/harmonizer 2>/dev/null || echo "checkout: clean"
```

## Secrets

Removing the credential **file** and revoking the **tokens** inside it are
two different things — do the first, not the second.

- The file (`~/.harmonizer.env` by default) survives every step above,
  because it sits outside the checkout by design. Deleting the checkout does
  not touch it. Remove it explicitly, as in step 2, or it stays on disk in
  plaintext.
- The tokens themselves are per-user and shared with your other work. Do
  **not** revoke `HF_TOKEN` or `NGC_API_KEY` as part of routine teardown.
  Rotate one only if you suspect it was printed, committed, or otherwise
  leaked.

Use length-only checks such as `${#HF_TOKEN}`; never echo token values.
