"""Experimento orientado a objetos: datos, MLP, entrenamiento y evaluación."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42


def fijar_semilla(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class DatosFrijol:
    """Carga, limpia, divide y normaliza los datos."""
    def __init__(self, ruta):
        self.ruta = Path(ruta); self.df = None; self.X = None; self.y = None
        self.clases = None; self.encoder = LabelEncoder(); self.scaler = StandardScaler()
        self.train_loader = self.val_loader = self.test_loader = None

    def preparar(self):
        original = pd.read_excel(self.ruta)
        if "Class" not in original or len(original.columns) - 1 != 16:
            raise ValueError(f"Se esperaban 16 características y Class: {list(original.columns)}")
        print("Dimensiones originales:", original.shape)
        print("Duplicados:", int(original.duplicated().sum()))
        self.df = original.drop_duplicates().reset_index(drop=True)
        print("Dimensiones limpias:", self.df.shape)
        carpeta = self.ruta.parents[1] / "processed"
        carpeta.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(carpeta / "datos_limpios.csv", index=False)

        # X son las características y y es la clase que se predice.
        self.X = self.df.drop(columns="Class")
        self.y = self.encoder.fit_transform(self.df["Class"])
        self.clases = list(self.encoder.classes_)

        # Primero se aparta test; después se obtiene validation del resto.
        indices = np.arange(len(self.df))
        train_idx, test_idx = train_test_split(indices, test_size=.15,
                                                random_state=SEED, stratify=self.y)
        train_idx, val_idx = train_test_split(train_idx, test_size=.1764705882,
                                              random_state=SEED, stratify=self.y[train_idx])
        assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx))
        print(f"Train: {len(train_idx)} | Validation: {len(val_idx)} | Test: {len(test_idx)}")
        print(pd.DataFrame({"train": pd.Series(self.y[train_idx]).value_counts().sort_index(),
                            "validation": pd.Series(self.y[val_idx]).value_counts().sort_index(),
                            "test": pd.Series(self.y[test_idx]).value_counts().sort_index()},
                           index=range(len(self.clases))).fillna(0).astype(int))
        self.df.iloc[train_idx].to_csv(carpeta / "datos_train.csv", index=False)
        self.df.iloc[val_idx].to_csv(carpeta / "datos_validation.csv", index=False)
        self.df.iloc[test_idx].to_csv(carpeta / "datos_test.csv", index=False)

        # El scaler se ajusta solo con train para evitar fuga de información.
        X_train = self.scaler.fit_transform(self.X.iloc[train_idx])
        X_val = self.scaler.transform(self.X.iloc[val_idx])
        X_test = self.scaler.transform(self.X.iloc[test_idx])
        self.train_loader = self._loader(X_train, self.y[train_idx], True)
        self.val_loader = self._loader(X_val, self.y[val_idx], False)
        self.test_loader = self._loader(X_test, self.y[test_idx], False)

    @staticmethod
    def _loader(features, labels, shuffle):
        dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
                                torch.tensor(labels, dtype=torch.long))
        return DataLoader(dataset, batch_size=64, shuffle=shuffle)


class MLP(nn.Module):
    """Red neuronal con arquitectura 16-64-32-7."""
    def __init__(self, entradas, salidas):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(entradas, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, salidas))

    def forward(self, entradas):
        return self.red(entradas)


class Entrenador:
    """Realiza entrenamiento, validación y parada temprana."""
    def __init__(self, modelo, dispositivo):
        self.modelo = modelo.to(dispositivo); self.dispositivo = dispositivo
        self.perdida = nn.CrossEntropyLoss()
        self.optimizador = torch.optim.Adam(self.modelo.parameters(), lr=.001)
        self.mejor_estado = None; self.mejor_epoca = 0; self.mejor_perdida = float("inf")

    def _paso(self, cargador, entrenar):
        self.modelo.train(entrenar); error_total = aciertos = total = 0
        for features, labels in cargador:
            features, labels = features.to(self.dispositivo), labels.to(self.dispositivo)
            if entrenar: self.optimizador.zero_grad()
            with torch.set_grad_enabled(entrenar):
                logits = self.modelo(features); error = self.perdida(logits, labels)
                if entrenar: error.backward(); self.optimizador.step()
            error_total += error.item() * len(labels)
            aciertos += (logits.argmax(1) == labels).sum().item(); total += len(labels)
        return error_total / total, aciertos / total

    def entrenar(self, datos, epocas=100, paciencia=15):
        historial = []; espera = 0
        for epoca in range(1, epocas + 1):
            train_loss, train_acc = self._paso(datos.train_loader, True)
            val_loss, val_acc = self._paso(datos.val_loader, False)
            historial.append({"epoch": epoca, "train_loss": train_loss, "val_loss": val_loss,
                              "train_accuracy": train_acc, "val_accuracy": val_acc})
            if val_loss < self.mejor_perdida - 1e-7:
                self.mejor_perdida = val_loss; self.mejor_estado = copy.deepcopy(self.modelo.state_dict())
                self.mejor_epoca = epoca; espera = 0
            else: espera += 1
            if epoca == 1 or epoca % 10 == 0:
                print(f"Época {epoca}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if espera >= paciencia:
                print("Parada temprana en época", epoca); break
        self.modelo.load_state_dict(self.mejor_estado)
        return pd.DataFrame(historial)


class Evaluador:
    """Calcula métricas de test y guarda el modelo."""
    def __init__(self, entrenador, datos, salida):
        self.entrenador = entrenador; self.datos = datos; self.salida = Path(salida)
        self.metricas = self.salida / "metricas"; self.modelos = self.salida / "modelos"
        self.metricas.mkdir(parents=True, exist_ok=True); self.modelos.mkdir(parents=True, exist_ok=True)

    def evaluar(self, historial):
        modelo = self.entrenador.modelo; modelo.eval(); pred, real = [], []; suma = 0; total = 0
        with torch.no_grad():
            for features, labels in self.datos.test_loader:
                logits = modelo(features.to(self.entrenador.dispositivo))
                error = self.entrenador.perdida(logits, labels.to(self.entrenador.dispositivo))
                suma += error.item() * len(labels); total += len(labels)
                pred.extend(logits.argmax(1).cpu().numpy()); real.extend(labels.numpy())
        pred, real = np.array(pred), np.array(real)
        reporte = classification_report(real, pred, labels=np.arange(len(self.datos.clases)),
                                        target_names=self.datos.clases, output_dict=True, zero_division=0)
        matriz = confusion_matrix(real, pred, labels=np.arange(len(self.datos.clases)))
        resumen = {"test_loss": suma / total, "accuracy": accuracy_score(real, pred),
                   "precision_macro": reporte["macro avg"]["precision"], "recall_macro": reporte["macro avg"]["recall"],
                   "f1_macro": reporte["macro avg"]["f1-score"], "f1_weighted": reporte["weighted avg"]["f1-score"],
                   "best_epoch": self.entrenador.mejor_epoca, "best_val_loss": self.entrenador.mejor_perdida}
        historial.to_csv(self.metricas / "historial_entrenamiento.csv", index=False)
        pd.DataFrame(reporte).T.to_csv(self.metricas / "metricas_por_clase.csv")
        pd.DataFrame([resumen]).to_csv(self.metricas / "resumen_metricas.csv", index=False)
        pd.DataFrame(matriz, index=self.datos.clases, columns=self.datos.clases).to_csv(self.metricas / "matriz_confusion.csv")
        pd.DataFrame({"real": [self.datos.clases[i] for i in real], "prediccion": [self.datos.clases[i] for i in pred]}).to_csv(self.metricas / "predicciones_test.csv", index=False)
        torch.save({"model_state_dict": modelo.state_dict(), "classes": self.datos.clases,
                    "scaler_mean": self.datos.scaler.mean_, "scaler_scale": self.datos.scaler.scale_},
                   self.modelos / "mejor_modelo_mlp.pt")
        print("Accuracy test:", f"{resumen['accuracy']:.2%}"); print("F1 macro:", f"{resumen['f1_macro']:.2%}")
        return {"history": historial, "summary": resumen, "report": pd.DataFrame(reporte).T,
                "confusion_matrix": matriz, "labels": self.datos.clases, "device": str(self.entrenador.dispositivo)}


def ejecutar_experimento(data_path, output_root, epochs=100, patience=15):
    """Coordina los objetos del experimento."""
    fijar_semilla(); datos = DatosFrijol(data_path); datos.preparar()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entrenador = Entrenador(MLP(len(datos.X.columns), len(datos.clases)), dispositivo)
    historial = entrenador.entrenar(datos, epochs, patience)
    return Evaluador(entrenador, datos, output_root).evaluar(historial)
