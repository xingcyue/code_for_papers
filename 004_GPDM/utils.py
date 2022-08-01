import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision.utils import save_image
from tqdm import tqdm


def get_first_initial_guess(reference_images, init_from, additive_noise_sigma):
    synthesized_images = torch.zeros_like(reference_images)
    if init_from == "zeros":
        print("reshuffle")
    elif init_from == "target":
        print("retarget")
        synthesized_images = reference_images.clone()
        import torchvision
        synthesized_images = torchvision.transforms.GaussianBlur(7, sigma=7)(synthesized_images)
    elif os.path.exists(init_from):
        print("style")
        synthesized_images = transforms.ToTensor()(Image.open(init_from)).unsqueeze(0) * 2.0 - 1.0
    if additive_noise_sigma:
        synthesized_images += torch.randn_like(synthesized_images) * additive_noise_sigma

    return synthesized_images


def get_pyramid_scales(max_height, min_height, pyramid_factor):
    cur_scale = max_height
    scales = [cur_scale]
    while cur_scale > min_height:
        cur_scale = int(cur_scale * pyramid_factor)
        scales.append(cur_scale)

    return scales[::-1]


def dump_images(images, out_dir):
    if os.path.exists(out_dir):
        i = len(os.listdir(out_dir))
    else:
        i = 0
        os.makedirs(out_dir)
    for j in range(images.shape[0]):
        save_image(images[j], os.path.join(out_dir, f"{i}.png"), normalize=True)
        i += 1


def get_output_shape(h, w, size, aspect_ratio):
    """Get the size of the output pyramid level"""
    h, w = int(size * aspect_ratio[0]), int((w * size / h) * aspect_ratio[1])
    return h, w


def match_patch_distributions(synthesized_images, reference_images, criteria, num_steps, lr):
    synthesized_images.requires_grad_(True)
    optim = torch.optim.Adam([synthesized_images], lr=lr)
    for i in tqdm(range(num_steps)):
        # Optimize image
        optim.zero_grad()
        loss = criteria(synthesized_images, reference_images)
        loss.backward()
        optim.step()

    return torch.clip(synthesized_images.detach(), -1, 1)
