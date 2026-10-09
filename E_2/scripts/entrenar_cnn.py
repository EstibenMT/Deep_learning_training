"""Entrenamiento y selección del checkpoint CNN mediante validación."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from datos_fashion import CargadorFashionMNIST
from modelo_cnn import CNNFashion

SEED = 42
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def fijar_semilla():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class EntrenadorCNN:
    def __init__(self, model, device, criterion, optimizer):
        self.model, self.device = model.to(device), device
        self.criterion, self.optimizer = criterion, optimizer

    def paso(self, loader, training):
        self.model.train(training)
        total_loss = correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            if training: self.optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                logits = self.model(images); loss = self.criterion(logits, labels)
                if training: loss.backward(); self.optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item(); total += len(labels)
        return total_loss / total, correct / total

    def fit(self, loaders, epochs=20, patience=5):
        history, best_state, best_loss, best_epoch, wait = [], None, float("inf"), 0, 0
        start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.paso(loaders["train"], True)
            val_loss, val_acc = self.paso(loaders["val"], False)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "train_accuracy": train_acc, "val_accuracy": val_acc})
            if val_loss < best_loss - 1e-6:
                best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(self.model.state_dict()), epoch, 0
            else: wait += 1
            print(f"Época {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4%} | val_loss={val_loss:.4f} | val_acc={val_acc:.4%}")
            if wait >= patience:
                print(f"Parada temprana en época {epoch}"); break
        self.model.load_state_dict(best_state)
        elapsed = time.perf_counter() - start
        return pd.DataFrame(history), best_epoch, best_loss, elapsed


def ejecutar_entrenamiento(proyecto: Path, epochs=20, patience=5):
    fijar_semilla(); proyecto = Path(proyecto)
    npz = proyecto / "data" / "raw" / "fashion_mnist_openml.npz"
    output = proyecto / "outputs"; metricas = output / "metricas"; modelos = output / "modelos"; config_dir = output / "configuracion"
    for folder in (metricas, modelos, config_dir): folder.mkdir(parents=True, exist_ok=True)
    data = CargadorFashionMNIST(npz, batch_size=128)
    loaders = {"train": data.train_loader(), "val": data.val_loader(), "test": data.test_loader()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNFashion(num_classes=10)
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = EntrenadorCNN(model, device, criterion, optimizer)
    print(model); print("Feature shape:", model.feature_shapes()); print("Dispositivo:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    history, best_epoch, best_val_loss, elapsed = trainer.fit(loaders, epochs, patience)
    history.to_csv(metricas / "historial_entrenamiento.csv", index=False)
    checkpoint = modelos / "mejor_modelo_cnn.pt"
    torch.save({"model_state_dict": model.state_dict(), "classes": CLASS_NAMES,
                "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                "architecture": str(model)}, checkpoint)
    config = {"seed": SEED, "epochs_max": epochs, "patience": patience, "batch_size": 128,
              "learning_rate": 0.001, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "best_epoch": best_epoch, "best_val_loss": best_val_loss, "training_seconds": elapsed}
    with (config_dir / "configuracion_entrenamiento.json").open("w", encoding="utf-8") as file: json.dump(config, file, indent=2, ensure_ascii=False)
    np.savez(metricas / "indices_particiones.npz", train=data.train_idx, validation=data.val_idx, test=data.test_idx)
    return {"data": data, "loaders": loaders, "model": model, "device": device, "history": history, "checkpoint": checkpoint, "config": config}
