# Clasificación de variedades de fríjol seco con MLP

Proyecto académico del Momento Evaluativo I de Introducción al Aprendizaje Profundo. Se utiliza el Dry Bean Dataset de UCI y un perceptrón multicapa implementado en PyTorch para clasificar siete variedades.

## Ejecución

Desde la raíz del repositorio:

```powershell
& .\.venv\Scripts\python.exe .\E_1\scripts\descargar_dataset.py
& .\.venv\Scripts\jupyter.exe notebook .\E_1\notebooks\Clasificacion_Dry_Bean_MLP_PyTorch.ipynb
```

También puede abrirse el notebook desde Jupyter y ejecutarse con el kernel `Python Deep Learning`. La primera celda encuentra automáticamente la carpeta `E_1`, aunque el notebook se abra desde la raíz, `E_1` o `E_1/notebooks`.

El código está separado por responsabilidad:

- `scripts/entrenar_mlp.py`: carga, limpieza, división, normalización, modelo, entrenamiento y métricas.
- `scripts/visualizar_resultados.py`: genera las figuras a partir de los resultados guardados.
- `scripts/ejecutar_proyecto.py`: ejecuta primero el entrenamiento y después las visualizaciones.

Para ejecutar todo sin abrir el notebook:

```powershell
& .\.venv\Scripts\python.exe .\E_1\scripts\ejecutar_proyecto.py
```

El dataset debe estar en `E_1/data/raw/Dry_Bean_Dataset.xlsx`. La descarga usa únicamente la fuente oficial de UCI.

Los datos preparados se guardan en `E_1/data/processed/`: `datos_limpios.csv`, `datos_train.csv`, `datos_validation.csv` y `datos_test.csv`.

## Salidas

El notebook guarda figuras en `E_1/outputs/figuras/`, métricas CSV en `E_1/outputs/metricas/` y el mejor modelo en `E_1/outputs/modelos/mejor_modelo_mlp.pt`.

## Orden recomendado

1. Descargar o verificar el dataset.
2. Ejecutar todas las celdas del notebook de arriba abajo.
3. Revisar las cantidades de train, validation y test.
4. Analizar las métricas por clase, el F1 macro y las matrices de confusión.
5. Redactar las conclusiones únicamente con los resultados generados.

Fuente del dataset: [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry%2Bbean%2Bdataset). Referencias técnicas: [PyTorch](https://pytorch.org/docs/stable/index.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) y [métricas de clasificación de scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
