"""
Dataset for breast ultrasound / mammogram images with segmentation masks.
Supports the BUSI dataset format (https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
or any folder with structure:

    data_root/
        benign/
            image_001.png
            image_001_mask.png
        malignant/
            image_001.png
            image_001_mask.png
        normal/        (optional - treated as benign with empty mask)
            ...

If no data is found, falls back to a synthetic dataset useful for smoke-testing
the full training and inference pipeline.
"""
import os
import glob
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _is_mask(path):
    name = os.path.basename(path).lower()
    return "_mask" in name or name.endswith("_mask.png") or name.endswith("_mask.jpg")


class BreastCancerDataset(Dataset):
    """Reads image-mask pairs and returns tensors ready for the model."""

    def __init__(self, root, img_size=224, train=True):
        self.root = root
        self.img_size = img_size
        self.train = train
        self.samples = []  # list of (image_path, mask_path_or_None, label)

        if root is not None and Path(root).exists():
            self._scan_real_dataset(root)

        if not self.samples:
            print(f"[Dataset] No real data found at {root!r} — using synthetic data.")
            self._make_synthetic(n=200)

    # ------------------------------------------------------------------ scan
    def _scan_real_dataset(self, root):
        class_to_label = {"benign": 0, "normal": 0, "malignant": 1}
        for cls_name, label in class_to_label.items():
            cls_dir = Path(root) / cls_name
            if not cls_dir.exists():
                continue
            images = sorted(
                p for p in glob.glob(str(cls_dir / "*"))
                if not _is_mask(p) and p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )
            for img_path in images:
                stem = Path(img_path).stem
                # try common mask naming conventions
                candidates = [
                    cls_dir / f"{stem}_mask.png",
                    cls_dir / f"{stem}_mask.jpg",
                    cls_dir / f"{stem}_mask.bmp",
                    cls_dir / f"{stem}.mask.png",
                ]
                mask_path = next((str(c) for c in candidates if c.exists()), None)
                self.samples.append((img_path, mask_path, label))

    # ------------------------------------------------------------ synthetic
    def _make_synthetic(self, n=200):
        """
        Build an in-memory synthetic dataset:
            benign  -> small, smooth circular hyperechoic blob
            malignant -> larger, irregular, spiculated hypoechoic blob
        Useful purely as a placeholder so the pipeline runs end-to-end.
        """
        self.synthetic = []
        for i in range(n):
            label = i % 2  # alternate
            img, mask = self._render_synthetic_sample(label)
            self.synthetic.append((img, mask, label))
        # samples list is used by __len__; align it
        self.samples = self.synthetic

    def _render_synthetic_sample(self, label):
        s = self.img_size
        # noisy ultrasound-like background
        bg = np.random.normal(60, 25, (s, s)).clip(0, 255).astype(np.uint8)
        bg = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(2)).convert("RGB")
        mask = Image.new("L", (s, s), 0)

        cx, cy = random.randint(60, s - 60), random.randint(60, s - 60)
        if label == 1:  # malignant: irregular & spiculated
            n_pts = 14
            radius = random.randint(35, 55)
            pts = []
            for k in range(n_pts):
                ang = 2 * np.pi * k / n_pts
                rr = radius * random.uniform(0.55, 1.4)
                pts.append((cx + rr * np.cos(ang), cy + rr * np.sin(ang)))
            ImageDraw.Draw(mask).polygon(pts, fill=255)
            ImageDraw.Draw(bg).polygon(pts, fill=(30, 30, 30))
        else:  # benign: smooth oval
            r = random.randint(18, 30)
            ImageDraw.Draw(mask).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
            ImageDraw.Draw(bg).ellipse((cx - r, cy - r, cx + r, cy + r), fill=(180, 180, 180))

        bg = bg.filter(ImageFilter.GaussianBlur(0.6))
        return bg, mask

    # --------------------------------------------------------------- length
    def __len__(self):
        return len(self.samples)

    # ----------------------------------------------------------- transforms
    def _augment(self, img, mask):
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if self.train and random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        return img, mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        # Real-data tuple is (path, path_or_None, label); synthetic is (PIL, PIL, label)
        if isinstance(item[0], str):
            img = Image.open(item[0]).convert("RGB").resize((self.img_size, self.img_size))
            if item[1] is not None:
                mask = Image.open(item[1]).convert("L").resize(
                    (self.img_size, self.img_size), Image.NEAREST
                )
            else:
                mask = Image.new("L", (self.img_size, self.img_size), 0)
        else:
            img, mask = item[0], item[1]

        img, mask = self._augment(img, mask)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        mask_t = (TF.to_tensor(mask) > 0.5).float()  # binarise
        label_t = torch.tensor(item[2], dtype=torch.long)
        return img_t, mask_t, label_t


def denormalize(img_tensor):
    """Reverse ImageNet normalization for visualisation."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)
