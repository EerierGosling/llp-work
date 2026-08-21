# Princeton LLP - Visual AI Lab

This repo contains some of my work during my internship at the Princeton Visual AI Lab through the Princeton LLP program!

I explored the connections between the gradients of diffusion models and adversarial classifiers. Our hypothesis was that noise augmentation in both model types might produce similar gradient patterns, since diffusion models denoise images step by step while adversarial classifiers are trained to resist small input perturbations. If it works, this could point toward distilling a classifier straight out of a diffusion model! I trained multiple adversarial classifiers on CIFAR-10 with epsilon values from 0.01 to 0.3 using PGD and FGSM attacks, then compared their input gradients to a pretrained diffusion model's at various noise levels.
