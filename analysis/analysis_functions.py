import torchvision.models as models
import torch.nn as nn
import torch
import glob
from types import SimpleNamespace
from generate_data import *
import os

def run_classifer(image, label, adversarial, epsilon=0, device='cuda', resnet_type='resnet34'):
    model = models.resnet34()
    model.maxpool = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    if epsilon == 0:
        epsilon = "0"

    model_path = f'/n/fs/visualai-scr/temp_LLP/sofia/llp-work/trained-models/{"adversarial" if adversarial else "non_adversarial"}/{resnet_type}/{epsilon}/2025-08-04_14-42-25.pth'

    model.load_state_dict(torch.load(model_path))

    model.eval()
    model = model.to(device)

    input_batch = image.unsqueeze(0)
    input_batch = input_batch.to(device)
    input_batch.requires_grad = True

    output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    predicted_class = predicted_idx.item()

    score = output[0, label]

    gradients = torch.autograd.grad(outputs=score, inputs=input_batch)[0]

    saliency = torch.abs(gradients)
    saliency_map = torch.max(saliency, dim=1)[0].squeeze().cpu().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

    return gradients, saliency_map, predicted_class


def run_diffusion(input_image, label, timestep, device='cuda'):
    # Specify the configuration
    args =  SimpleNamespace(
        dataset='cifar10',
        timesteps=1000,
        device=device,
        batch_size=1,
        guidance_scale=2.0,
        ddim=True,
        sampling_steps=50,
        pretrained_ckpt='/n/fs/wy-project/minimal-diffusion/trained_models3/UNet_cifar10-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt',
        arch='UNet',
        diffusion_steps=1000,
    )

    diffusion = GaussianDiffusion(args.diffusion_steps, args.device)

    # setup diffusion model
    metadata = get_metadata(args.dataset)
    model = unets.__dict__[args.arch](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes,
    ).to(args.device)

    d = fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
    model.load_state_dict(d, strict=False)

    # Sample from the diffusion model
    model.eval()

    cond_gradients = sample_image(
        model,
        diffusion,
        input_image.to(args.device),
        label,
        timestep,
        args=args
    )

    normalized_gradients = cond_gradients[0]
    normalized_gradients = (normalized_gradients - normalized_gradients.min()) / (normalized_gradients.max() - normalized_gradients.min() + 1e-8)

    saliency_diffusion = np.abs(cond_gradients)
    saliency_map_diffusion = np.max(saliency_diffusion, axis=0)
    saliency_map_diffusion = (saliency_map_diffusion - saliency_map_diffusion.min()) / (saliency_map_diffusion.max() - saliency_map_diffusion.min() + 1e-8)

    return cond_gradients[0], saliency_map_diffusion

def mean_squared_error(gradient1, gradient2, device='cpu'):
    if isinstance(gradient1, np.ndarray):
        gradient1 = torch.from_numpy(gradient1)
    if isinstance(gradient2, np.ndarray):
        gradient2 = torch.from_numpy(gradient2)

    gradient1 = gradient1.to(device)
    gradient2 = gradient2.to(device)

    while gradient1.dim() > 3:
        gradient1 = gradient1.squeeze(0)
    while gradient2.dim() > 3:
        gradient2 = gradient2.squeeze(0)

    gradient1 = (gradient1 - gradient1.min()) / (gradient1.max() - gradient1.min() + 1e-8)
    gradient2 = (gradient2 - gradient2.min()) / (gradient2.max() - gradient2.min() + 1e-8)

    return torch.mean((gradient1 - gradient2) ** 2).cpu().item()

def cosine_similarity(gradient1, gradient2, device='cpu'):
    if isinstance(gradient1, np.ndarray):
        gradient1 = torch.from_numpy(gradient1)
    if isinstance(gradient2, np.ndarray):
        gradient2 = torch.from_numpy(gradient2)

    gradient1 = gradient1.to(device)
    gradient2 = gradient2.to(device)

    while gradient1.dim() > 3:
        gradient1 = gradient1.squeeze(0)
    while gradient2.dim() > 3:
        gradient2 = gradient2.squeeze(0)
    
    cos_sim = torch.nn.functional.cosine_similarity(gradient1.unsqueeze(0), gradient2.unsqueeze(0), dim=1)
    return cos_sim.mean().cpu().item()