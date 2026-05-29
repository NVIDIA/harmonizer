# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Training script that mirrors the legacy
`train_pix2pix_turbo_nocond_cosmos_skip_tp_multidataset_optical.py`:
  * supports multiple datasets (comma-separated --dataset_folder)
  * weighted sampler across datasets
  * RAFT optical-flow temporal loss
  * 5D (B, C, V, H, W) tensor handling, V==1 included.
"""

import gc
import os
import random
from glob import glob

#import clip
import diffusers
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

from opticalflow_loss import get_temporal_loss, get_temporal_loss_model
from pix2pix_turbo_harmonizer import (
    Pix2Pix_Turbo,
    load_ckpt_from_state_dict,
    save_ckpt,
)
from utils.style_loss import style_loss
from utils.training_utils import PairedDataset, PairedDatasetV2, parse_args_paired_training

LOSS_INPUT_ONLY = True


def build_train_val_datasets(args):
    """Per-source dataset list so we can build a WeightedRandomSampler."""
    sources = [s.strip() for s in args.dataset_folder.split(",") if s.strip()]
    train_list, val_list = [], []
    for source in sources:
        print(f"------- Processing source {source}")
        # JSON manifest sources use PairedDatasetV2; directory-style sources
        # (with optional `+ref_method` multiview suffix) use PairedDataset.
        if ".json" in source:
            ds_train = PairedDatasetV2(dataset_folder=source, image_prep=args.train_image_prep, split="train")
            ds_val = PairedDatasetV2(dataset_folder=source, image_prep=args.test_image_prep, split="test")
        else:
            ds_train = PairedDataset(dataset_folder=source, image_prep=args.train_image_prep, split="train")
            ds_val = PairedDataset(dataset_folder=source, image_prep=args.test_image_prep, split="test")
        random.Random(42).shuffle(ds_val.img_names)
        train_list.append(ds_train)
        val_list.append(ds_val)
    return train_list, val_list


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_pix2pix = Pix2Pix_Turbo(
        experiment_name=args.experiment_name,
        s3_checkpoint_dir=args.s3_checkpoint_dir,
        freeze_vae_encoder=args.freeze_vae_encoder,
        freeze_vae=args.freeze_vae,
        train_full_unet=args.train_full_unet,
        timestep=args.timestep,
        use_sched=args.use_sched,
        vae_skip_connection=args.vae_skip_connection,
        hf_path=args.hf_path,
        pretrained_path=args.pretrained_path,
    )
    net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if not args.swinir:
            if is_xformers_available():
                net_pix2pix.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    use_gan = args.lambda_gan > 0
    use_clipsim = args.lambda_clipsim > 0

    if use_gan:
        if args.gan_disc_type == "vagan_clip":
            import vision_aided_loss

            net_disc = vision_aided_loss.Discriminator(cv_type="clip", loss_type=args.gan_loss_type, device="cuda")
        else:
            raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

        net_disc = net_disc.cuda()
        net_disc.requires_grad_(True)
        net_disc.cv_ensemble.requires_grad_(False)
        net_disc.train()
    else:
        net_disc = None

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    if use_clipsim:
        net_clip, _ = clip.load("ViT-B/32", device="cuda")
        net_clip.requires_grad_(False)
        net_clip.eval()
    else:
        net_clip = None
    net_lpips.requires_grad_(False)

    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for p in net_vgg.parameters():
        p.requires_grad_(False)

    layers_to_opt = []
    if args.train_full_unet:
        print("=" * 50, "\nadding unet parameters\n", "=" * 50)
        layers_to_opt += list(net_pix2pix.unet.parameters())
    if not args.freeze_vae:
        if args.freeze_vae_encoder:
            print("------- adding vae decoder parameters")
            layers_to_opt += list(net_pix2pix._inner_vae().decoder.parameters())
        else:
            print("------- adding whole vae parameters")
            layers_to_opt += list(net_pix2pix._inner_vae().parameters())

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if use_gan:
        optimizer_disc = torch.optim.AdamW(
            net_disc.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_disc = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    else:
        optimizer_disc = None
        lr_scheduler_disc = None

    # -------- Datasets / Dataloaders --------
    dataset_train_list, dataset_val_list = build_train_val_datasets(args)
    dataset_train = torch.utils.data.ConcatDataset(dataset_train_list)
    dataset_val = torch.utils.data.ConcatDataset(dataset_val_list)

    if args.weighted_sampler:
        sources = [s.strip() for s in args.dataset_folder.split(",") if s.strip()]
        lengths = [len(ds) for ds in dataset_train_list]
        weights = []
        for src, l in zip(sources, lengths):
            source_mult = args.fixing_data_weight if "nre_data" in src else 1.0
            print(f"------- Source '{src}' length={l} sampler multiplier={source_mult}")
            w = source_mult / l
            weights.extend([w] * l)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=sum(lengths), replacement=True
        )
        dl_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.train_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
        )
    else:
        dl_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
        )

    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # -------- Resume --------
    global_step = 0
    if args.resume is not None:
        if os.path.isdir(args.resume):
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(
                ckpt_files,
                key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")),
            )
            print("=" * 50, f"\nLoading checkpoint from {ckpt_files[-1]}\n", "=" * 50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_pix2pix, net_disc, optimizer, optimizer_disc = load_ckpt_from_state_dict(
                net_pix2pix, net_disc, optimizer, optimizer_disc, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("=" * 50, f"\nLoading checkpoint from {args.resume}\n", "=" * 50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_pix2pix, net_disc, optimizer, optimizer_disc = load_ckpt_from_state_dict(
                net_pix2pix, net_disc, optimizer, optimizer_disc, args.resume
            )
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("=" * 50, "\nTraining from scratch\n", "=" * 50)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    if use_gan:
        net_disc.to(accelerator.device, dtype=weight_dtype)
    if use_clipsim:
        net_clip.to(accelerator.device, dtype=weight_dtype)

    if use_gan:
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
            net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
        )
    else:
        net_pix2pix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
            net_pix2pix, optimizer, dl_train, lr_scheduler
        )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)
    if use_clipsim:
        net_clip = accelerator.prepare(net_clip)

    t_clip_renorm = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )
    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if accelerator.is_main_process:
        init_kwargs = {"wandb": {"name": args.tracker_run_name, "dir": args.output_dir}}
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)), init_kwargs=init_kwargs)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    if use_gan:
        for name, module in net_disc.named_modules():
            if "attn" in name:
                module.fused_attn = False

    # RAFT temporal-loss helpers
    raft, temp_loss = get_temporal_loss_model(accelerator, weight_dtype, use_small=False)

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc] if use_gan else [net_pix2pix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                if len(x_src.shape) == 4:
                    x_src = rearrange(x_src, "b (c v) h w -> b v c h w", v=1)
                    x_tgt = rearrange(x_tgt, "b (c v) h w -> b v c h w", v=1)

                assert len(x_src.shape) == 5 and len(x_tgt.shape) == 5
                B, V, C, H, W = x_src.shape

                x_src = rearrange(x_src, "b v c h w -> b c v h w")
                x_tgt = rearrange(x_tgt, "b v c h w -> b c v h w")

                x_tgt_pred = net_pix2pix(x_src)

                if V != 1 and args.lambda_opticalflow > 0:
                    loss_tmporal = get_temporal_loss(
                        x_tgt_pred=x_tgt_pred.float(),
                        x_tgt=x_tgt.float(),
                        raft=raft,
                        temp_loss=temp_loss,
                    ) * args.lambda_opticalflow
                else:
                    loss_tmporal = torch.tensor(0.0, device=x_tgt_pred.device, dtype=x_tgt_pred.dtype)

                x_tgt_visual = x_tgt.detach().clone()
                if LOSS_INPUT_ONLY:
                    x_tgt = x_tgt[:, :, :1]
                    x_tgt_pred = x_tgt_pred[:, :, :1]

                x_tgt = rearrange(x_tgt, "b c v h w -> (b v) c h w")
                x_tgt_pred = rearrange(x_tgt_pred, "b c v h w -> (b v) c h w")

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                lpips_val = net_lpips(x_tgt_pred.float(), x_tgt.float())
                # Clamp the loss to regularize the supervision
                lpips_val = lpips_val.clamp(0, 0.3)
                loss_lpips = lpips_val.mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips + loss_tmporal

                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h, crop_w = 512, 512
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)

                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)

                        loss_gram = (
                            style_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg)
                            * args.lambda_gram
                        )
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)

                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(
                        x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False
                    )
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = 1 - clipsim.mean() / 100
                    loss += loss_clipsim * args.lambda_clipsim

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if LOSS_INPUT_ONLY:
                    x_tgt = rearrange(x_tgt, "(b v) c h w -> b c v h w", v=1)
                    x_tgt_pred = rearrange(x_tgt_pred, "(b v) c h w -> b c v h w", v=1)
                else:
                    x_tgt = rearrange(x_tgt, "(b v) c h w -> b c v h w", v=V)
                    x_tgt_pred = rearrange(x_tgt_pred, "(b v) c h w -> b c v h w", v=V)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    if args.lambda_gan > 0:
                        logs["lossG"] = lossG.detach().item()  # noqa: F821
                        logs["lossD"] = lossD.detach().item()  # noqa: F821
                    logs["loss_l2"] = loss_l2.detach().item()
                    if isinstance(loss_tmporal, torch.Tensor) and loss_tmporal.detach().item() != 0:
                        logs["loss_tmporal"] = loss_tmporal.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 1:
                        if V > 1 and args.lambda_opticalflow > 0:
                            _, visual_warp = get_temporal_loss(
                                x_tgt_pred, x_tgt_visual, raft, temp_loss, visual=True
                            )
                            log_dict = {
                                "train/x_src": [
                                    wandb.Image(rearrange(x_src, "b c v h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}")
                                    for idx in range(B)
                                ],
                                "train/x_tgt_visual": [
                                    wandb.Image(rearrange(x_tgt_visual, "b c v h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}")
                                    for idx in range(B)
                                ],
                                "train/x_tgt_pred": [
                                    wandb.Image(rearrange(x_tgt_pred, "b c v h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}")
                                    for idx in range(B)
                                ],
                                "train/visual_warp": [
                                    wandb.Image(rearrange(visual_warp, "b c v h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}")
                                    for idx in range(B)
                                ],
                            }
                        else:
                            log_dict = {
                                "train/x_src": [
                                    wandb.Image(rearrange(x_src, "b c v h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}")
                                    for idx in range(B)
                                ],
                                "train/x_tgt_visual": [
                                    wandb.Image(rearrange(x_tgt_visual, "b c v h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}")
                                    for idx in range(B)
                                ],
                                "train/x_tgt_pred": [
                                    wandb.Image(rearrange(x_tgt_pred, "b c v h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}")
                                    for idx in range(B)
                                ],
                            }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        save_ckpt(
                            accelerator.unwrap_model(net_pix2pix),
                            net_disc,
                            optimizer,
                            optimizer_disc,
                            outf,
                            train_full_unet=args.train_full_unet,
                            freeze_vae=args.freeze_vae,
                        )

                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for vstep, batch_val in enumerate(dl_val):
                            if vstep >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            if len(x_src.shape) == 4:
                                x_src = rearrange(x_src, "b (c v) h w -> b v c h w", v=1)
                                x_tgt = rearrange(x_tgt, "b (c v) h w -> b v c h w", v=1)
                            B, V, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                x_src = rearrange(x_src, "b v c h w -> b c v h w")
                                x_tgt = rearrange(x_tgt, "b v c h w -> b c v h w")
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src)

                                if vstep % 10 == 0:
                                    log_dict["sample/source"].append(
                                        wandb.Image(rearrange(x_src, "b c v h w -> b c (v h) w")[0].float().detach().cpu(),
                                                    caption=f"idx={len(log_dict['sample/source'])}")
                                    )
                                    log_dict["sample/target"].append(
                                        wandb.Image(rearrange(x_tgt, "b c v h w -> b c (v h) w")[0].float().detach().cpu(),
                                                    caption=f"idx={len(log_dict['sample/source'])}")
                                    )
                                    log_dict["sample/model_output"].append(
                                        wandb.Image(rearrange(x_tgt_pred, "b c v h w -> b c (v h) w")[0].float().detach().cpu(),
                                                    caption=f"idx={len(log_dict['sample/source'])}")
                                    )

                                if LOSS_INPUT_ONLY:
                                    x_tgt = x_tgt[:, :, :1]
                                    x_tgt_pred = x_tgt_pred[:, :, :1]

                                x_tgt = rearrange(x_tgt, "b c v h w -> (b v) c h w")
                                x_tgt_pred = rearrange(x_tgt_pred, "b c v h w -> (b v) c h w")

                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())

                                if use_clipsim:
                                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                                    x_tgt_pred_renorm = F.interpolate(
                                        x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False
                                    )
                                    caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                    l_clipsim.append(clipsim.mean().item())

                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].float().cpu() * 0.5 + 0.5)
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{vstep}.png")
                                output_pil.save(outf)

                        logs["val/l2"] = float(np.mean(l_l2)) if l_l2 else 0.0
                        logs["val/lpips"] = float(np.mean(l_lpips)) if l_lpips else 0.0
                        logs["val/clipsim"] = float(np.mean(l_clipsim)) if l_clipsim else 0.0
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
