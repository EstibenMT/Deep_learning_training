"""Punto de entrada: primero entrena y después genera las visualizaciones."""

from pathlib import Path

from entrenar_mlp import ejecutar_experimento
from visualizar_resultados import generar_graficas


def main() -> None:
    proyecto = Path(__file__).resolve().parents[1]
    data_path = proyecto / "data" / "raw" / "Dry_Bean_Dataset.xlsx"
    output_root = proyecto / "outputs"
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {data_path}")

    # Fase 1: datos, modelo, entrenamiento y métricas.
    resultado = ejecutar_experimento(data_path, output_root)
    # Fase 2: figuras para analizar los resultados.
    generar_graficas(data_path, output_root)
    print("Accuracy test:", f"{resultado['summary']['accuracy']:.2%}")
    print("F1 macro:", f"{resultado['summary']['f1_macro']:.2%}")


if __name__ == "__main__":
    main()
