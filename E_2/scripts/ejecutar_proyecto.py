"""Punto de entrada de la Entrega 2."""

from pathlib import Path

from descargar_fashion_mnist import preparar_dataset
from entrenar_cnn import ejecutar_entrenamiento
from evaluar_cnn import evaluar
from visualizar_cnn import generar_figuras


def main():
    proyecto = Path(__file__).resolve().parents[1]
    preparar_dataset(proyecto)
    resultado = ejecutar_entrenamiento(proyecto, epochs=20, patience=5)
    generar_figuras(proyecto, resultado)
    evaluar(resultado, proyecto)


if __name__ == "__main__":
    main()
