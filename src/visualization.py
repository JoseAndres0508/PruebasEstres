"""Módulo de Visualización.

Genera las gráficas de T(x), L(x) y E(x) con el área bajo la curva
sombreada (representando el AUC calculado en src/integration.py) y la
tabla comparativa entre la Regla del Trapecio y la Regla de Simpson 1/3.

Las funciones devuelven objetos matplotlib.figure.Figure en lugar de
llamar a plt.show(): así quedan listas para integrarse sin cambios
dentro del dashboard (TG7), usando
matplotlib.backends.backend_tkagg.FigureCanvasTkAgg. Mientras tanto,
este módulo también puede guardarlas como imágenes independientes (ver
guardar_figura), útiles para el informe y los anexos.

Corresponde a la Tarea Global TG5.

Responsable: Jose Andrés Ortiz Marín
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, para poder guardar figuras desde consola
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Configuración visual de cada variable: título, etiqueta del eje Y y color.
CONFIG_VARIABLES = {
    "T": {"titulo": "Throughput T(x)", "ylabel": "Throughput (sol/seg)", "color": "steelblue"},
    "L": {"titulo": "Latencia L(x)", "ylabel": "Latencia (ms)", "color": "darkorange"},
    "E": {"titulo": "Tasa de error E(x)", "ylabel": "Tasa de error (%)", "color": "crimson"},
}


def graficar_variable(
    x: np.ndarray,
    y: np.ndarray,
    titulo: str,
    ylabel: str,
    color: str = "steelblue",
) -> Figure:
    """Grafica una curva de rendimiento y sombrea el área bajo la curva.

    El sombreado (fill_between) representa visualmente el AUC calculado
    numéricamente en src/integration.py.

    Args:
        x: usuarios concurrentes.
        y: valores de la variable evaluada en cada x.
        titulo: título de la gráfica.
        ylabel: etiqueta del eje Y (incluir unidad).
        color: color de la curva y del área sombreada.

    Returns:
        Figure de Matplotlib lista para embeberse en el dashboard o
        guardarse como imagen.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, color=color, linewidth=2, label=titulo)
    ax.fill_between(x, y, color=color, alpha=0.25, label="Área bajo la curva (AUC)")
    ax.set_title(titulo)
    ax.set_xlabel("Usuarios concurrentes (x)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def graficar_todas_las_variables(datos: pd.DataFrame) -> dict[str, Figure]:
    """Genera una figura por variable (T, L, E), con el AUC sombreado.

    Args:
        datos: DataFrame con columnas ["x", "T", "L", "E"] (ver
            src.simulation.generar_datos_simulacion).

    Returns:
        Diccionario {"T": Figure, "L": Figure, "E": Figure}.
    """
    x = datos["x"].to_numpy()
    figuras = {}
    for variable, config in CONFIG_VARIABLES.items():
        y = datos[variable].to_numpy()
        figuras[variable] = graficar_variable(x, y, config["titulo"], config["ylabel"], config["color"])
    return figuras


def tabla_comparativa_auc(resultados_auc: pd.DataFrame) -> Figure:
    """Genera una tabla comparando el AUC por Regla del Trapecio y Simpson 1/3.

    Args:
        resultados_auc: DataFrame producido por
            src.integration.calcular_auc(), con columnas [variable,
            unidad, auc_trapecio, auc_simpson, diferencia_%].

    Returns:
        Figure de Matplotlib con la tabla renderizada, lista para
        embeberse en el dashboard o guardarse como imagen.
    """
    columnas = ["Variable", "Unidad", "AUC Trapecio", "AUC Simpson", "Diferencia %"]
    filas = [
        [
            fila["variable"],
            fila["unidad"],
            f"{fila['auc_trapecio']:,.4f}",
            f"{fila['auc_simpson']:,.4f}",
            f"{fila['diferencia_%']:.4f}%",
        ]
        for _, fila in resultados_auc.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.5 * len(filas)))
    ax.axis("off")
    tabla = ax.table(cellText=filas, colLabels=columnas, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)
    for j in range(len(columnas)):
        celda = tabla[(0, j)]
        celda.set_facecolor("#1F4E79")
        celda.set_text_props(color="white", weight="bold")
    fig.tight_layout()
    return fig


def guardar_figura(fig: Figure, ruta: str, dpi: int = 150) -> None:
    """Guarda una figura de Matplotlib en disco.

    Args:
        fig: figura a guardar.
        ruta: ruta del archivo de salida (ej. "assets/grafica_throughput.png").
        dpi: resolución de la imagen.
    """
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    from src.integration import calcular_auc
    from src.simulation import generar_datos_simulacion

    # n=101 (impar): requisito de Simpson 1/3, ver src/integration.py
    datos = generar_datos_simulacion(n=101)
    resultados_auc = calcular_auc(datos)

    figuras = graficar_todas_las_variables(datos)
    guardar_figura(figuras["T"], "assets/grafica_throughput.png")
    guardar_figura(figuras["L"], "assets/grafica_latencia.png")
    guardar_figura(figuras["E"], "assets/grafica_tasa_error.png")

    tabla_fig = tabla_comparativa_auc(resultados_auc)
    guardar_figura(tabla_fig, "assets/tabla_comparativa_auc.png")

    print("Gráficas guardadas en assets/:")
    print("  grafica_throughput.png")
    print("  grafica_latencia.png")
    print("  grafica_tasa_error.png")
    print("  tabla_comparativa_auc.png")
    print()
    print("=== Tabla comparativa AUC (Trapecio vs. Simpson 1/3) ===")
    print(resultados_auc.to_string(index=False))

    plt.close("all")