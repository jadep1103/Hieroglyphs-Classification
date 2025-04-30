import os
from PIL import Image
import PIL
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
import numpy as np
from math import ceil
from collections import defaultdict


def generate_noise_background(size, mean=0.72, std=0.07):
    noise = np.random.normal(loc=mean, scale=std, size=size).clip(0, 1)
    return torch.tensor(noise, dtype=torch.float32)


def augment_image(pil_img, n=5):
    augmented_imgs = []
    for _ in range(n):

        img_aug = augmentations(pil_img)
        img_tensor = transforms.ToTensor()(img_aug)

        noise = generate_noise_background(img_tensor.shape)

        mask = (img_tensor < 0.083)
        img_tensor = torch.where(mask, noise, img_tensor)

        img_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(img_tensor)

        augmented_imgs.append(img_tensor)
    return augmented_imgs

if __name__ == "__main__":
    augmentations = transforms.Compose([
        transforms.RandomAffine(
            degrees=16,
            translate=(0.15, 0.22),
            scale=(0.65, 1),
            shear=5,
            fill = 0,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((416, 416), interpolation=transforms.InterpolationMode.BICUBIC)
    ])

    input_root = "EgyptianHieroglyphDataset-1/train"
    output_root = "EgyptianHieroglyphDataset-1/train_aug"
    target_per_class = 200
    max_aug_per_image = 15
    os.makedirs(output_root, exist_ok=True)

    class_counts = defaultdict(int)
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                rel_dir = os.path.relpath(dirpath, input_root)
                class_counts[rel_dir] += 1

    for dirpath, _, filenames in os.walk(input_root):
        rel_dir = os.path.relpath(dirpath, input_root)
        n_images = class_counts[rel_dir]
        if n_images == 0:
            continue

        aug_per_image = max(0, ceil((target_per_class - n_images) / n_images))
        aug_per_image = min(aug_per_image, max_aug_per_image)

        for filename in tqdm(filenames, desc=f"{rel_dir}"):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            input_path = os.path.join(dirpath, filename)
            output_dir = os.path.join(output_root, rel_dir)
            os.makedirs(output_dir, exist_ok=True)

            if aug_per_image == 0:
                try:
                    img = Image.open(input_path).convert("L")
                except (IOError, OSError, PIL.UnidentifiedImageError) as e:
                    continue
                img.save(os.path.join(output_dir, filename))
            if aug_per_image > 0:
                try:
                    img = Image.open(input_path).convert("L")
                except (IOError, OSError, PIL.UnidentifiedImageError) as e:
                    continue
                img.save(os.path.join(output_dir, filename))
                augmented_imgs = augment_image(img, aug_per_image)
                base_name = Path(filename).stem
                for i, aug_img in enumerate(augmented_imgs):
                    img_tensor = aug_img * 0.5 + 0.5
                    img_pil = transforms.ToPILImage()(img_tensor)
                    aug_name = f"{base_name}_aug_{i}.png"
                    img_pil.save(os.path.join(output_dir, aug_name))