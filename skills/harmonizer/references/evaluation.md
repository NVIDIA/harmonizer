# DiffusionHarmonizer — Quantitative evaluation

> **This workflow does not currently run as shipped.** Two blockers live in
> the repository's own code, not in this skill:
>
> - `src/evaluate_test_dataset.py:109-117` calls `load_and_compile_model(...)`
>   without the `image_size` argument its signature requires
>   (`src/inference_pretrained_model.py:386-396`).
> - It loads the fast-tokenizer pipeline, whose config
>   (`src/pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py:29-35`) expects
>   `/work/models/base/model_fast_tokenizer.pt` and
>   `/work/models/base/tokenizer_fast.pth`. `download_checkpoints.sh` does not
>   fetch either — it provides `models/diffusion_harmonizer.pkl`,
>   `models/harmonizer_nontemporal.pt`, and the standard Cosmos model under
>   `src/checkpoints/`.
>
> Treat everything below as the intended shape rather than a working recipe,
> and do not promise a user that evaluation will complete. Raise both with the
> harmonizer maintainers before relying on this path.

For PSNR / LPIPS evaluation, prepare paired data with this
layout:

```text
test_dataset/
├── {scene_id_1}/
│   ├── render/
│   │   ├── {camera_id_1}/
│   │   │   ├── {timestamp_1}.png
│   │   │   └── {timestamp_2}.png
│   │   └── {camera_id_2}/
│   └── gt/
│       ├── {camera_id_1}/
│       │   ├── {timestamp_1}.png
│       │   └── {timestamp_2}.png
│       └── {camera_id_2}/
└── {scene_id_2}/
    ├── render/
    └── gt/
```

`render/` and `gt/` must have identical camera subdirectories
and matching filenames. Images may be PNG, JPEG, or JPG.

## Run evaluation

The shape below is the **intended** invocation, not a working recipe — see
the warning at the top of this file.

Note in particular that the base-model path is not what it appears: the
imported fast-tokenizer pipeline looks for
`/work/models/base/model_fast_tokenizer.pt` and
`/work/models/base/tokenizer_fast.pth`, **not** the `src/checkpoints/` model
`download_checkpoints.sh` provides. That mismatch is one of the two blockers.

Mount the whole checkout at `/work` and run from `/work/src`. Place the test
dataset under `/work` as well (or mount it read-write so the script can write
`evaluation/` and `metrics.yaml`):

```bash
# the harmonizer checkout you built the image from
CODE_DIR=$(cd /absolute/path/to/harmonizer && pwd)
```

```bash
docker run --gpus=all --rm --ipc=host \
  -u "$(id -u):$(id -g)" \
  -v "$CODE_DIR":/work \
  -v /absolute/path/to/test_dataset:/work/test_dataset \
  -w /work/src \
  --entrypoint python \
  harmonizer-cosmos-env \
  evaluate_test_dataset.py \
    --model /work/models/diffusion_harmonizer.pkl \
    --input /work/test_dataset \
    --output /work/test_dataset/evaluation
```

Add `--calculate-for-input` to also report metrics between the raw
input renders and ground truth (in addition to enhanced output vs. GT).

## Expected outputs

- Enhanced images under the `--output` directory (default
  `evaluation/`) that mirrors the test dataset structure.
- `metrics.yaml` in the output directory with overall and per-scene
  PSNR/LPIPS, inference time, image counts, and GPU memory statistics.
