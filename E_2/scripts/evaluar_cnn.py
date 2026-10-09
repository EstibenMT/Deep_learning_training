"""Evaluación final, métricas, ROC-AUC y ejemplos de predicciones."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_fscore_support)

from entrenar_cnn import CLASS_NAMES


def evaluar(resultado, proyecto: Path):
    output = Path(proyecto) / "outputs"; metricas = output / "metricas"; ejemplos = output / "ejemplos"
    figuras = output / "figuras"; ejemplos.mkdir(parents=True, exist_ok=True); figuras.mkdir(parents=True, exist_ok=True)
    model, device, loader = resultado["model"], resultado["device"], resultado["loaders"]["test"]
    model.eval(); criterion = torch.nn.CrossEntropyLoss(); loss_sum = 0; total = 0; y_true = []; y_pred = []; probabilities = []; images_all = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)); loss = criterion(logits, labels.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            loss_sum += loss.item() * len(labels); total += len(labels)
            y_true.extend(labels.numpy()); y_pred.extend(logits.argmax(1).cpu().numpy()); probabilities.extend(probs); images_all.extend(images.numpy())
    y_true, y_pred, probabilities, images_all = map(np.asarray, (y_true, y_pred, probabilities, images_all))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    report = classification_report(y_true, y_pred, labels=np.arange(10), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(metricas / "metricas_por_clase.csv")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(metricas / "matriz_confusion.csv")
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(metricas / "matriz_confusion_normalizada.csv")
    one_vs_rest = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        one_vs_rest.append({"class": name, "specificity": tn / (tn + fp) if tn + fp else 0, "false_positive_rate": fp / (fp + tn) if fp + tn else 0, "false_negative_rate": fn / (fn + tp) if fn + tp else 0})
    pd.DataFrame(one_vs_rest).to_csv(metricas / "metricas_one_vs_rest.csv", index=False)
    y_onehot = np.eye(10)[y_true]
    auc_values = {CLASS_NAMES[i]: roc_auc_score(y_onehot[:, i], probabilities[:, i]) for i in range(10)}
    auc_values["macro_auc"] = roc_auc_score(y_onehot, probabilities, multi_class="ovr", average="macro")
    pd.DataFrame([auc_values]).to_csv(metricas / "roc_auc.csv", index=False)
    summary = {"test_loss": loss_sum / total, "accuracy": float((y_true == y_pred).mean()), "f1_macro": report["macro avg"]["f1-score"], "f1_weighted": report["weighted avg"]["f1-score"], "macro_auc": auc_values["macro_auc"], "best_epoch": resultado["config"]["best_epoch"], "best_val_loss": resultado["config"]["best_val_loss"]}
    pd.DataFrame([summary]).to_csv(metricas / "resumen_metricas.csv", index=False)
    _save_confusion(cm, cm_norm, figuras); _save_roc(y_onehot, probabilities, figuras); _save_examples(images_all, y_true, y_pred, probabilities, ejemplos)
    print("Test loss:", f"{summary['test_loss']:.4f}"); print("Test accuracy:", f"{summary['accuracy']:.2%}"); print("F1 macro:", f"{summary['f1_macro']:.2%}"); print("ROC-AUC macro:", f"{summary['macro_auc']:.4f}")
    return summary


def _save_confusion(cm, cm_norm, folder):
    for matrix, title, name, fmt in [(cm, "Matriz de confusión", "matriz_confusion.png", "d"), (cm_norm, "Matriz de confusión normalizada", "matriz_confusion_normalizada.png", ".2f")]:
        fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax); ax.set(xlabel="Predicción", ylabel="Clase real", title=title); fig.tight_layout(); fig.savefig(folder / name, dpi=150); plt.close(fig)


def _save_roc(y_onehot, probabilities, folder):
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], probabilities[:, i]); ax.plot(fpr, tpr, label=f"{name}")
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio"); ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC one-vs-rest"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(folder / "roc_auc_por_clase.png", dpi=150); plt.close(fig)


def _save_examples(images, true, pred, probabilities, folder):
    correct = np.where(true == pred)[0][:5]; incorrect = np.where(true != pred)[0][:5]
    for indices, name, title in [(correct, "correctas.png", "Predicciones correctas"), (incorrect, "incorrectas.png", "Predicciones incorrectas")]:
        fig, axes = plt.subplots(1, len(indices), figsize=(15, 3)); axes = np.atleast_1d(axes)
        for ax, index in zip(axes, indices):
            ax.imshow(images[index, 0], cmap="gray"); ax.set_title(f"R: {CLASS_NAMES[true[index]]}\nP: {CLASS_NAMES[pred[index]]}\n{probabilities[index, pred[index]]:.1%}"); ax.axis("off")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(folder / name, dpi=150); plt.close(fig)
