"""Pruebas del Módulo de Visualización (TG5)."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from src.integration import calcular_auc
from src.simulation import generar_datos_simulacion
from src.visualization import (
    graficar_todas_las_variables,
    graficar_variable,
    guardar_figura,
    tabla_comparativa_auc,
)


@pytest.fixture
def datos_muestra() -> pd.DataFrame:
    return generar_datos_simulacion(n=51)


@pytest.fixture
def auc_muestra(datos_muestra: pd.DataFrame) -> pd.DataFrame:
    return calcular_auc(generar_datos_simulacion(n=101))  # n impar, requisito de Simpson 1/3


def test_graficar_variable_retorna_figura():
    x = np.linspace(10, 500, 50)
    y = np.linspace(1, 480, 50)
    fig = graficar_variable(x, y, "Prueba", "unidad (u)")
    assert isinstance(fig, Figure)


def test_graficar_variable_sombrea_area_bajo_la_curva():
    x = np.linspace(0, 10, 20)
    y = x ** 2
    fig = graficar_variable(x, y, "Prueba", "unidad (u)")
    ax = fig.axes[0]
    # fill_between agrega un PolyCollection a las colecciones del eje.
    assert len(ax.collections) >= 1


def test_graficar_variable_titulo_y_etiquetas_correctos():
    x = np.array([1, 2, 3])
    y = np.array([1, 4, 9])
    fig = graficar_variable(x, y, "Mi título", "Mi eje Y (ms)")
    ax = fig.axes[0]
    assert ax.get_title() == "Mi título"
    assert ax.get_ylabel() == "Mi eje Y (ms)"
    assert ax.get_xlabel() == "Usuarios concurrentes (x)"


def test_graficar_todas_las_variables_retorna_las_tres_figuras(datos_muestra):
    figuras = graficar_todas_las_variables(datos_muestra)
    assert set(figuras.keys()) == {"T", "L", "E"}
    assert all(isinstance(fig, Figure) for fig in figuras.values())


def test_tabla_comparativa_auc_retorna_figura_con_tabla(auc_muestra):
    fig = tabla_comparativa_auc(auc_muestra)
    ax = fig.axes[0]
    assert isinstance(fig, Figure)
    assert len(ax.tables) == 1


def test_tabla_comparativa_auc_tiene_una_fila_por_variable(auc_muestra):
    fig = tabla_comparativa_auc(auc_muestra)
    ax = fig.axes[0]
    tabla = ax.tables[0]
    # +1 por la fila de encabezados (Variable, Unidad, AUC Trapecio, ...)
    filas_esperadas = len(auc_muestra) + 1
    filas_en_tabla = len({celda[0] for celda in tabla.get_celld().keys()})
    assert filas_en_tabla == filas_esperadas


def test_guardar_figura_crea_archivo(tmp_path):
    x = np.linspace(0, 10, 20)
    y = x ** 2
    fig = graficar_variable(x, y, "Prueba", "unidad (u)")
    ruta = tmp_path / "figura_prueba.png"

    guardar_figura(fig, str(ruta))

    assert ruta.exists()
    assert ruta.stat().st_size > 0