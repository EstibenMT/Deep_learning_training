"""Visualizaciones del dataset y del historial de la CNN."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from entrenar_cnn import CLASS_NAMES


def generar_figuras(proyecto: Path, resultado: dict):
    proyecto = Path(proyecto); figures = proyecto / "outputs" / "figuras"; figures.mkdir(parents=True, exist_ok=True)
    data = resultado["data"]
    # Distribución y ejemplos reales del conjunto de entrenamiento.
    labels = data.train_dataset.labels
    fig, ax = plt.subplots(figsize=(10, 5)); sns.countplot(x=[CLASS_NAMES[i] for i in labels], order=CLASS_NAMES, ax=ax); ax.tick_params(axis="x", rotation=35); ax.set_title("Distribución de clases en entrenamiento"); fig.tight_layout(); fig.savefig(figures / "distribucion_clases.png", dpi=150); plt.close(fig)
    indices = [np.where(labels == i)[0][0] for i in range(10)]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, (ax, idx) in enumerate(zip(axes.ravel(), indices)):
        image, _ = data.train_dataset[idx]; ax.imshow(image[0], cmap="gray"); ax.set_title(CLASS_NAMES[i]); ax.axis("off")
    fig.suptitle("Ejemplo de cada clase"); fig.tight_layout(); fig.savefig(figures / "ejemplos_clases.png", dpi=150); plt.close(fig)
    history = resultado["history"]
    for train_col, val_col, title, name in [("train_loss", "val_loss", "Pérdida", "curvas_perdida.png"), ("train_accuracy", "val_accuracy", "Accuracy", "curvas_accuracy.png")]:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(history.epoch, history[train_col], label="Entrenamiento"); ax.plot(history.epoch, history[val_col], label="Validación"); ax.set(xlabel="Época", ylabel=title, title=f"{title}: entrenamiento vs validación"); ax.legend(); fig.tight_layout(); fig.savefig(figures / name, dpi=150); plt.close(fig)
    print("Figuras guardadas en:", figures)
