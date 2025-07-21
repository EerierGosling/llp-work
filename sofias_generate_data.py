import os
import cv2
import copy
import math
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data import get_metadata, get_dataset, fix_legacy_dict
import unets

unsqueeze3x = lambda x: x[..., None, None, None]


class GaussianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, t2=None, model_kwargs={}, ddim=False, guidance_scale=2.0
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.
        guidance_scale: Controls the strength of classifier-free guidance (1.0 means no guidance)

        Return: An image tensor with identical shape as XT.
        """
        assert "y" in model_kwargs, "Class labels are required for class-conditioned models"
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            
            if t != t2:
                continue

            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                
                # For classifier-free guidance, we need both conditional and unconditional predictions
                    # Get unconditional prediction (using label=0)
                unconditional_kwargs = model_kwargs.copy()
                unconditional_kwargs["y"] = torch.zeros_like(model_kwargs["y"])
                unconditional_pred = model(final, current_t, **unconditional_kwargs)
                
                # Get conditional prediction
                conditional_pred = model(final, current_t, **model_kwargs)
                
                cond_score = (unconditional_pred - conditional_pred) / unsqueeze3x((scalars.beta_tilde[current_sub_t].sqrt()).sqrt())
                    
        return cond_score


class loss_logger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.loss = []
        self.start_time = time()
        self.ema_loss = None
        self.ema_w = 0.9

    def log(self, v, display=False):
        self.loss.append(v)
        if self.ema_loss is None:
            self.ema_loss = v
        else:
            self.ema_loss = self.ema_w * self.ema_loss + (1 - self.ema_w) * v

        if display:
            print(
                f"Steps: {len(self.loss)}/{self.max_steps} \t loss (ema): {self.ema_loss:.3f} "
                + f"\t Time elapsed: {(time() - self.start_time)/3600:.3f} hr"
            )


def sample_image(
    model,
    diffusion,
    input_image,
    input_label,
    timestep,
    sampling_steps=250,
    args=None,
):
    """use this function to sample images from a given diffusion model and diffusion process
       within a specific index range.

    Args:
        model : Diffusion model
        diffusion : Diffusion process
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with images and corresponding labels.
    """
    
    noisy_image, _ = diffusion.sample_from_forward_process(input_image, timestep)
    noisy_image = noisy_image.unsqueeze(0).to(args.device)

    y = torch.tensor([input_label + 1], device=args.device)
    
    cond_gradients = diffusion.sample_from_reverse_process(
        model, noisy_image, sampling_steps, {"y": y}, args.ddim, args.guidance_scale
    )
            
    return cond_gradients.cpu().numpy()