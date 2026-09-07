"""Genera las gráficas a partir de los datos y resultados guardados."""

from pathlib import Path

import matplotlib

# Permite guardar figuras desde la terminal sin abrir ventanas gráficas.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generar_graficas(data_path: Path, output_root: Path) -> None:
    """Crea y guarda las figuras del informe."""
    output_root = Path(output_root)
    figuras = output_root / "figuras"
    metricas = output_root / "metricas"
    figuras.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(data_path).drop_duplicates()

    # Distribución de las clases.
    orden = df["Class"].value_counts().index
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=df, x="Class", order=orden, ax=axes[0])
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Registros por clase")
    porcentajes = df["Class"].value_counts(normalize=True).reindex(orden) * 100
    sns.barplot(x=porcentajes.index, y=porcentajes.values, ax=axes[1])
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Porcentaje por clase")
    axes[1].set_ylabel("Porcentaje (%)")
    fig.tight_layout()
    fig.savefig(figuras / "distribucion_clases.png", dpi=150)
    plt.close(fig)

    # Correlación entre las características numéricas.
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlación entre características")
    fig.tight_layout()
    fig.savefig(figuras / "correlaciones.png", dpi=150)
    plt.close(fig)

    # Histogramas de cuatro características representativas.
    columnas = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, columna in zip(axes.ravel(), columnas):
        sns.histplot(data=df, x=columna, hue="Class", element="step",
                     stat="density", common_norm=False, ax=ax)
        ax.set_title(f"Distribución de {columna}")
    fig.tight_layout()
    fig.savefig(figuras / "histogramas.png", dpi=150)
    plt.close(fig)

    # Curvas de pérdida y accuracy.
    history = pd.read_csv(metricas / "historial_entrenamiento.csv")
    for train_col, val_col, title, name in [
        ("train_loss", "val_loss", "Curvas de pérdida", "curvas_perdida.png"),
        ("train_accuracy", "val_accuracy", "Curvas de accuracy", "curvas_accuracy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history.epoch, history[train_col], label="Entrenamiento")
        ax.plot(history.epoch, history[val_col], label="Validación")
        ax.set(xlabel="Época", ylabel="Valor", title=title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figuras / name, dpi=150)
        plt.close(fig)

    # Matrices de confusión absoluta y normalizada.
    matrix = pd.read_csv(metricas / "matriz_confusion.csv", index_col=0)
    for values, title, name, fmt in [
        (matrix, "Matriz de confusión", "matriz_confusion.png", "d"),
        (matrix.div(matrix.sum(axis=1), axis=0), "Matriz de confusión normalizada",
         "matriz_confusion_normalizada.png", ".2f"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(values, annot=True, fmt=fmt, cmap="Blues", ax=ax)
        ax.set(xlabel="Predicción", ylabel="Real", title=title)
        fig.tight_layout()
        fig.savefig(figuras / name, dpi=150)
        plt.close(fig)

    print("Gráficas guardadas en:", figuras)
