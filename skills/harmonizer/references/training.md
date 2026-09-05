# DiffusionHarmonizer — Dataset, training, and NuRec pair recipes

Detail moved out of `SKILL.md`. Run only when the user explicitly
asks for training, fine-tuning, or evaluation-pair construction.

## Dataset

The released dataset holds approximately **1.394 million** curated
synthetic-real image pairs from five complementary curation pipelines.

> **Check your free space first — this is ~1.76 TB.**
> The 180 published files total about 1.76 TB, and they are 5–70 GB tar
> archives that still need extracting. The single largest pair,
> `ISP_modification/train_A.tar` + `train_B.tar`, is ~137 GB on its own
> before extraction. The ~120 GB figure quoted for this skill covers
> **inference only** and does **not** cover the dataset; the prerequisite
> check does not verify space for it either.
>
> **~1.8 TB is the download alone — it is not enough to also extract.** Plan
> by how you unpack:
>
> - *Extract and delete each archive in turn:* archive footprint plus room
>   for the largest expanded archive — budget well above 1.8 TB, since the
>   biggest single tar is 68.5 GB before expansion.
> - *Keep the archives:* roughly download **plus** full extracted footprint,
>   i.e. on the order of 3.5 TB.
>
> Download only the subsets you need instead — see below.

Prefer downloading only the subsets you need with `--include`, rather than
the whole repository:

```bash
# one subset, not the full 1.76 TB
hf download nvidia/Harmonizer-Dataset \
  --repo-type dataset \
  --include 'shadow_PBR/*' \
  --local-dir data
```

`./download_checkpoints.sh --with-dataset` fetches the **entire** dataset
alongside the inference checkpoints. Do not use it unless you have ~1.8 TB
free for the download *and* the additional extraction space above.

Note the training code expects train/test prompt files or a generated JSON
manifest (`src/utils/training_utils.py:596-700`), and the sample command
below assumes `/data/data.json`. Converting the downloaded tarballs into
that layout is not covered here.

Data sources and targeted failure modes:

| Data source | Failure mode |
|-------------|--------------|
| ISP Modification | ISP-induced color or tone drift between foreground and background. |
| Relighting | Illumination mismatch between inserted objects and scene lighting. |
| Asset Re-insertion | Missing shadows and appearance mismatch when dynamic assets are re-inserted. |
| PBR Shadow Simulation | Missing or unrealistic cast shadows on inserted objects. |
| Artifacts Correction | Novel-view artifacts such as blur, missing regions, ghosting, and spurious geometry. |

Training JSON format:

```json
{
  "train": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "prompt": "remove degradation"
    }
  },
  "test": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "prompt": "remove degradation"
    }
  }
}
```

## Training

Recommended multi-GPU training shape from the release README.

**Run this from `src/`, not the repository root.**
`src/pix2pix_turbo_harmonizer.py:32` resolves the base model as
`checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/model.pt` relative to the
process working directory, and `download_checkpoints.sh` places it under
`src/checkpoints/...`. Launching from the repo root makes that path miss.

```bash
cd src            # or -w /work/src inside the container

export NUM_NODES=1
export NUM_GPUS=8
export OUTPUT_DIR=/path/to/checkpointing_directory
export DATASET_FOLDER=/data/data.json
export WANDB_MODE=offline

accelerate launch \
  --mixed_precision=bf16 \
  --main_process_port 29501 \
  --multi_gpu \
  --num_machines "$NUM_NODES" \
  --num_processes "$NUM_GPUS" \
  train_pix2pix_turbo_harmonizer.py \
    --output_dir="${OUTPUT_DIR}" \
    --dataset_folder="${DATASET_FOLDER}" \
    --max_train_steps 10000 \
    --learning_rate 2e-5 \
    --train_batch_size=1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --checkpointing_steps=2000 \
    --eval_freq 1000 \
    --viz_freq 1000 \
    --train_image_prep resize_576x1024 \
    --test_image_prep resize_576x1024 \
    --lambda_clipsim 0.0 \
    --lambda_lpips 0.3 \
    --lambda_gan 0.0 \
    --lambda_l2 1.0 \
    --lambda_gram 0.0 \
    --use_sched \
    --report_to wandb \
    --tracker_project_name cosmos_harmonizer \
    --tracker_run_name train \
    --train_full_unet \
    --timestep 250 \
    --track_val_fid \
    --num_samples_eval 20 \
    --mixed_precision=bf16
```

For fine-tuning from the released checkpoint, add:

```bash
--pretrained_path /path/to/diffusion_harmonizer.pkl
```

When omitted, the model is fine-tuned directly from the raw Cosmos 0.6B
image model.

The README recommends `--fixing_data_weight 3 --weighted_sampler` to
up-weight artifact-correction examples, but **it cannot work with the single
`/data/data.json` source shown above.** Per
`src/train_pix2pix_turbo_harmonizer.py:200-208` the sampler assigns one
weight per *source* and spreads it evenly across that source's items, so with
a single source the multiplier normalises out — even if the path contains
`nre_data`.

To actually up-weight, split the data and name the correction source so its
path contains `nre_data`:

```bash
--dataset_folder /data/base_data.json,/data/nre_data.json \
  --weighted_sampler \
  --fixing_data_weight 3
```

Without both the multi-source split and `--weighted_sampler`, omit the flag.

## NuRec data-pair recipes

The updated tutorials describe four ways to construct image pairs
for training or testing. Summarised below; the command templates
themselves are in the upstream `doc/dataset_preparation_tutorial.md`:

- **Sparse reconstruction:** train with every Nth frame and pair
  held-out ground-truth images with rendered novel views.
- **Cycle reconstruction:** listed as a supported pair-generation
  method in the tutorial overview.
- **Model underfitting:** train a reconstruction for a reduced
  schedule (roughly 25%-75%) and pair degraded renders with clean
  targets.
- **Cross reference:** train using one camera and render/evaluate
  held-out cameras.

When adapting the sample NuRec commands, keep public container
names and current NuRec recipes. Do not copy internal container
names or internal dataset paths into user-facing instructions.
