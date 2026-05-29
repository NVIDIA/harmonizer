# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from cosmos_predict2.conditioner import DataType, TextCondition
from cosmos_predict2.configs.base.config_text2image import Text2ImagePipelineConfig
from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
from cosmos_predict2.module.denoise_prediction import DenoisePrediction
from einops import rearrange
from imaginaire.lazy_config import LazyDict, instantiate
from imaginaire.utils import log

from utils import NVTXRangeDecorator

from .denoiser_scaling import RectifiedFlowScaling


class FastUnconditioner(torch.nn.Module):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def to_dict(self):
        return self.conditioner.to_dict()

    @staticmethod
    def sample_batch_image(batch_size: int, device: str | torch.device, dtype: torch.dtype, h: int = 544, w: int = 960):
        # h, w = 384, 640
        # h, w = 768, 1360
        # h, w = 544, 960
        data_batch = {
            "dataset_name": "image_data",
            "images": torch.zeros(batch_size, 3, h, w).to(device, dtype=dtype),
            "t5_text_embeddings": torch.zeros(batch_size, 512, 1024).to(device, dtype=dtype),
            "fps": torch.ones((batch_size,)).to(device, dtype=dtype) * 24,
            "padding_mask": torch.zeros(batch_size, 1, h, w).to(device, dtype=dtype),
        }
        return data_batch

    @classmethod
    def from_config(
        cls,
        config: LazyDict[Text2ImagePipelineConfig],
        batch_size: int,
        device: str,
        dtype: torch.dtype,
    ):
        pipe = cls(device=device, dtype=dtype)

        # Initialize conditioner
        conditioner = instantiate(config.conditioner)
        assert sum(p.numel() for p in conditioner.parameters() if p.requires_grad) == 0, (
            "conditioner should not have learnable parameters"
        )

        # Get uncondition
        data_batch = cls.sample_batch_image(batch_size, device=device, dtype=dtype)
        is_image_batch = True

        condition, uncondition = conditioner.get_condition_uncondition(data_batch)
        del condition

        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        pipe.conditioner = uncondition

        return pipe


class FastTokenizer(torch.nn.Module):
    def __init__(
        self,
        skip_connection=False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.skip_connection = skip_connection
        # assert self.skip_connection == False, "skip connection is not supported for now"

    @NVTXRangeDecorator("encode")
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state) * self.sigma_data

    @NVTXRangeDecorator("decode")
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.skip_connection:
            self.tokenizer.decoder.incoming_skip_acts = self.tokenizer.encoder.current_down_blocks
        return self.tokenizer.decode(latent / self.sigma_data)

    # def vae_encode(self, state: torch.Tensor) -> torch.Tensor:
    #     latents = self.vae.encode(state, self.scale)
    #     num_frames = latents.shape[2]
    #     if num_frames == 1:
    #         return (latents - self.img_mean.type_as(latents)) / self.img_std.type_as(latents)
    #     else:
    #         return (latents - self.video_mean[:, :, :num_frames].type_as(latents)) / self.video_std[
    #             :, :, :num_frames
    #         ].type_as(latents)

    # def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
    #     num_frames = latent.shape[2]
    #     if num_frames == 1:
    #         return self.vae.decode(
    #             (latent * self.img_std.type_as(latent)) + self.img_mean.type_as(latent), self.scale
    #         )
    #     else:
    #         return self.vae.decode(
    #             (latent * self.video_std[:, :, :num_frames].type_as(latent))
    #             + self.video_mean[:, :, :num_frames].type_as(latent), self.scale
    #         )

    def get_state_shape(self, H: int, W: int, _T: int = 1) -> tuple[int, int, int, int]:
        return (
            self.tokenizer.latent_ch,
            self.tokenizer.get_latent_num_frames(_T),
            H // self.tokenizer.spatial_compression_factor,
            W // self.tokenizer.spatial_compression_factor,
        )

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.tokenizer.load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_config(
        cls,
        config: LazyDict[Text2ImagePipelineConfig],
        skip_connection: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        pipe = cls(skip_connection=skip_connection, device=device, dtype=dtype)
        pipe.sigma_data = config.sigma_data
        # Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert pipe.tokenizer.latent_ch == config.state_ch, (
            f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {config.state_ch}"
        )
        return pipe


class FastDenoiser(torch.nn.Module):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def compute_caching(
        self,
        sigma: torch.Tensor,
        condition: TextCondition,
    ) -> None:
        """
        Because sigma and conditioning are constant, we can compute them ahead of time.
        """

        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")
        self.sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")

        # get precondition for the network
        self.c_skip_B_1_T_1_1, self.c_out_B_1_T_1_1, self.c_in_B_1_T_1_1, self.c_noise_B_1_T_1_1 = self.scaling(
            sigma=self.sigma_B_1_T_1_1
        )

        # compute t_embedding_B_T_D
        self.timesteps_B_T = self.c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
            **self.tensor_kwargs
        )  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf

        # compute crossattn_emb
        # NOTE:
        # To resolve the issue in onnx export:
        # Do NOT store `condition` on `self.condition`.
        #
        # In some call paths (e.g. Pix2Pix_Turbo init), `condition` may be a torch.nn.Module wrapper
        # (FastUnconditioner), which would register `condition` as a child module on first assignment.
        # Later assigning a non-Module (e.g. TextCondition) to the same attribute would raise:
        # "cannot assign ... as child module 'condition' (torch.nn.Module or None expected)".
        #
        # We only need the tensor fields from `condition.to_dict()` for caching.
        condition_dict = condition.to_dict()
        self.crossattn_emb = condition_dict["crossattn_emb"]
        self.data_type = condition_dict["data_type"]
        self.padding_mask = condition_dict["padding_mask"]
        self.fps = condition_dict["fps"]

        # propogate the caching to the DiT
        self.dit.compute_caching(timesteps_B_T=self.timesteps_B_T, crossattn_emb=self.crossattn_emb)

    @NVTXRangeDecorator("forward")
    def forward(self, xt_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass through the network
        """
        # forward pass through the network
        net_output_B_C_T_H_W = self.dit(
            x_B_C_T_H_W=(xt_B_C_T_H_W * self.c_in_B_1_T_1_1).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            fps=self.fps,
            padding_mask=self.padding_mask,
            data_type=self.data_type,
            use_cuda_graphs=False,
        ).float()
        x0_pred_B_C_T_H_W = self.c_skip_B_1_T_1_1 * xt_B_C_T_H_W + self.c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        return x0_pred_B_C_T_H_W

    @NVTXRangeDecorator("denoise")
    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        use_cuda_graphs: bool = False,
    ) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (TextCondition): conditional information, generated from self.conditioner
            use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """

        # forward pass through the network
        net_output_B_C_T_H_W = self.dit(
            x_B_C_T_H_W=(xt_B_C_T_H_W * self.c_in_B_1_T_1_1).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            fps=self.fps,
            padding_mask=self.padding_mask,
            data_type=self.data_type,
            use_cuda_graphs=use_cuda_graphs,
        ).float()

        x0_pred_B_C_T_H_W = self.c_skip_B_1_T_1_1 * xt_B_C_T_H_W + self.c_out_B_1_T_1_1 * net_output_B_C_T_H_W

        # get noise prediction
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / self.sigma_B_1_T_1_1

        return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None)

    @classmethod
    def from_config(
        cls,
        config: LazyDict[Text2ImagePipelineConfig],
        dit_path: str = "",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_ema_to_reg: bool = False,
        distill_steps: int = 0,
    ):
        # Create a pipe
        pipe = cls(device=device, dtype=dtype)
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
        log.warning(f"precision {pipe.precision}")

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        pipe.scaling = RectifiedFlowScaling(
            config.sigma_data,
            config.rectified_flow_t_scaling_factor,
            config.rectified_flow_loss_weight_uniform,
        )
        pipe.distill_steps = distill_steps

        assert config.guardrail_config.enabled == False, "guardrail is not supported"

        # 6. Set up DiT
        if dit_path:
            log.info(f"Loading DiT from {dit_path}")
        else:
            log.warning("dit_path not provided, initializing DiT with random weights")
        with init_weights_on_device():
            pipe.dit = instantiate(config.net).eval()  # inference

        if dit_path:
            state_dict = load_state_dict(dit_path)
            prefix_to_load = "net_ema." if load_ema_to_reg else "net."
            # drop net. prefix
            state_dict_dit_compatible = dict()
            for k, v in state_dict.items():
                if k.startswith(prefix_to_load):
                    state_dict_dit_compatible[k[len(prefix_to_load) :]] = v
                else:
                    state_dict_dit_compatible[k] = v
            pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
            del state_dict, state_dict_dit_compatible
            log.success(f"Successfully loaded DiT from {dit_path}")

        # 6-2. Handle EMA
        assert config.ema.enabled == False, "ema is not supported"

        # Finalize the model
        pipe.dit = pipe.dit.to(device=device, dtype=dtype)
        torch.cuda.empty_cache()
        return pipe
