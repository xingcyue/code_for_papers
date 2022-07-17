import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.RandomCrop(256),
                            transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, transforms=trans):
        content_images = glob.glob((content_dir + '/*'))
        np.random.shuffle(content_images)
        style_images = glob.glob(style_dir + '/*')
        np.random.shuffle(style_images)
        self.images_pairs = list(zip(content_images, style_images))
        self.transforms = transforms

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_image, style_image = self.images_pairs[index]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
        return content_image, style_image
