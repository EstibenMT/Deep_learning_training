"""CNN propia para imágenes Fashion-MNIST de 28x28 píxeles."""

import torch
from torch import nn


class CNNFashion(nn.Module):
    """Tres bloques convolucionales y un clasificador denso."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def feature_shapes(self):
        with torch.no_grad():
            device = next(self.parameters()).device
            dummy = torch.zeros(1, 1, 28, 28, device=device)
            return self.features(dummy).shape
