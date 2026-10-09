"""Dataset PyTorch y lectura de la partición guardada."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FashionMNISTDataset(Dataset):
    """Devuelve cada imagen como tensor [1, 28, 28] sin data augmentation."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class CargadorFashionMNIST:
    """Carga el NPZ preparado y construye DataLoaders reproducibles."""

    def __init__(self, npz_path: Path, batch_size: int = 128):
        data = np.load(npz_path)
        images, labels = data["X"], data["y"]
        self.train_idx = data["train_idx"]
        self.val_idx = data["val_idx"]
        self.test_idx = data["test_idx"]
        self.train_dataset = FashionMNISTDataset(images[self.train_idx], labels[self.train_idx])
        self.val_dataset = FashionMNISTDataset(images[self.val_idx], labels[self.val_idx])
        self.test_dataset = FashionMNISTDataset(images[self.test_idx], labels[self.test_idx])
        self.batch_size = batch_size

    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
