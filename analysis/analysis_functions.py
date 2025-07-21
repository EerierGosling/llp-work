import torchvision.models as models
import torch.nn as nn
import torch
import glob
from types import SimpleNamespace
from generate_data import *
import os

def run_classifer(image, adversarial, epsilon=0, device='cuda', resnet_type='resnet34'):
    model = models.resnet34()
    model.maxpool = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    if epsilon == 0:
        epsilon = "0"

    model_folder = f'/n/fs/visualai-scr/temp_LLP/sofia/llp-work/trained-models/{"adversarial" if adversarial else "non_adversarial"}/{resnet_type}/{epsilon}'
    files = glob.glob(os.path.join(model_folder, "*.pth"))
    
    most_recent = max(files, key=os.path.getmtime)

    model.load_state_dict(torch.load(most_recent))

    model.eval()
    model = model.to(device)

    input_batch = image.unsqueeze(0)
    input_batch = input_batch.to(device)
    input_batch.requires_grad = True

    output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    predicted_class = predicted_idx.item()

    score = output[0, predicted_class]

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

def mean_squared_error(gradient1, gradient2):
    # Convert both to torch tensors on the same device
    if isinstance(gradient1, np.ndarray):
        gradient1 = torch.from_numpy(gradient1)
    if isinstance(gradient2, np.ndarray):
        gradient2 = torch.from_numpy(gradient2)
    
    # Ensure they're on the same device
    device = gradient1.device if gradient1.is_cuda else 'cpu'
    gradient1 = gradient1.to(device)
    gradient2 = gradient2.to(device)
    
    # Flatten and ensure same shape
    gradient1_flat = gradient1.view(-1)
    gradient2_flat = gradient2.view(-1)
    
    # Take minimum length if they're different sizes
    min_len = min(gradient1_flat.size(0), gradient2_flat.size(0))
    gradient1_flat = gradient1_flat[:min_len]
    gradient2_flat = gradient2_flat[:min_len]
    
    return torch.mean((gradient1_flat - gradient2_flat) ** 2).cpu().item()

def cosine_similarity(gradient1, gradient2):
    # Convert both to torch tensors on the same device
    if isinstance(gradient1, np.ndarray):
        gradient1 = torch.from_numpy(gradient1)
    if isinstance(gradient2, np.ndarray):
        gradient2 = torch.from_numpy(gradient2)
    
    # Ensure they're on the same device
    device = gradient1.device if gradient1.is_cuda else 'cpu'
    gradient1 = gradient1.to(device)
    gradient2 = gradient2.to(device)
    
    # Flatten and ensure same shape
    gradient1_flat = gradient1.view(-1)
    gradient2_flat = gradient2.view(-1)
    
    # Take minimum length if they're different sizes
    min_len = min(gradient1_flat.size(0), gradient2_flat.size(0))
    gradient1_flat = gradient1_flat[:min_len]
    gradient2_flat = gradient2_flat[:min_len]
    
    # Reshape for cosine similarity
    gradient1_flat = gradient1_flat.unsqueeze(0)
    gradient2_flat = gradient2_flat.unsqueeze(0)
    
    cos_sim = torch.nn.functional.cosine_similarity(gradient1_flat, gradient2_flat, dim=1)
    return cos_sim.mean().cpu().item()