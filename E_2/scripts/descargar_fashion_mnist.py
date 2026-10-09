"""Descarga Fashion-MNIST desde OpenML y prepara los tres subconjuntos."""

from pathlib import Path
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def preparar_dataset(proyecto: Path | None = None) -> Path:
    proyecto = proyecto or Path(__file__).resolve().parents[1]
    raw = proyecto / "data" / "raw"
    processed = proyecto / "data" / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    archivo_npz = raw / "fashion_mnist_openml.npz"

    if archivo_npz.exists():
        print(f"Dataset local encontrado: {archivo_npz}")
        local = np.load(archivo_npz)
        hashes = [hashlib.sha1(image.tobytes()).hexdigest() for image in local["X"]]
        print(f"Duplicados exactos de imagen: {len(hashes) - len(Counter(hashes))}")
        return archivo_npz

    print("Descargando Fashion-MNIST desde OpenML (data_id=40996)...")
    dataset = fetch_openml(data_id=40996, as_frame=False, parser="auto", data_home=str(raw / "openml_cache"))
    X = np.asarray(dataset.data, dtype=np.uint8).reshape(-1, 28, 28)
    y = np.asarray(dataset.target, dtype=np.int64)
    if X.shape != (70000, 28, 28) or len(np.unique(y)) != 10:
        raise ValueError(f"Estructura inesperada: X={X.shape}, clases={np.unique(y)}")
    hashes = [hashlib.sha1(image.tobytes()).hexdigest() for image in X]
    duplicate_records = len(hashes) - len(Counter(hashes))
    print(f"Duplicados exactos de imagen: {duplicate_records}")

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=SEED, stratify=y)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1764705882, random_state=SEED, stratify=y[train_idx])
    assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx))

    np.savez_compressed(archivo_npz, X=X, y=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    pd.DataFrame({"indice_original": train_idx, "label": y[train_idx], "clase": [CLASS_NAMES[i] for i in y[train_idx]]}).to_excel(processed / "datos_train.xlsx", index=False)
    pd.DataFrame({"indice_original": val_idx, "label": y[val_idx], "clase": [CLASS_NAMES[i] for i in y[val_idx]]}).to_excel(processed / "datos_validation.xlsx", index=False)
    pd.DataFrame({"indice_original": test_idx, "label": y[test_idx], "clase": [CLASS_NAMES[i] for i in y[test_idx]]}).to_excel(processed / "datos_test.xlsx", index=False)
    pd.DataFrame({"label": np.arange(10), "clase": CLASS_NAMES}).to_csv(processed / "mapeo_clases.csv", index=False)
    with (processed / "resumen_dataset.json").open("w", encoding="utf-8") as salida:
        json.dump({"fuente": "OpenML data_id=40996", "total": int(len(y)), "dimensiones": [28, 28], "clases": CLASS_NAMES, "duplicados_exactos": int(duplicate_records), "semilla": SEED}, salida, indent=2, ensure_ascii=False)
    print(f"Dataset preparado en: {archivo_npz}")
    print(f"Train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    return archivo_npz


if __name__ == "__main__":
    preparar_dataset()
