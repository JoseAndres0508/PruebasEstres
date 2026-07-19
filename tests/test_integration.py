"""Pruebas del Módulo de Cálculo Integral Numérico (TG4)."""

import numpy as np
import pytest

from src.integration import (
    calcular_auc,
    simpson_1_3,
    trapecio,
    validar_precision,
)
from src.simulation import generar_datos_simulacion


def test_trapecio_es_exacto_para_funcion_lineal():
    # El Trapecio es exacto para cualquier función lineal (grado <= 1).
    x = np.linspace(0, 10, 11)
    y = 2 * x + 3
    exacta = ((2 * 10 + 3) + 3) / 2 * 10  # área del trapecio de f(0)=3 a f(10)=23
    assert trapecio(x, y) == pytest.approx(exacta, rel=1e-9)


def test_trapecio_valida_longitudes():
    with pytest.raises(ValueError):
        trapecio([1, 2, 3], [1, 2])


def test_trapecio_valida_minimo_de_puntos():
    with pytest.raises(ValueError):
        trapecio([1], [1])


def test_simpson_es_exacto_para_funcion_cubica():
    # Simpson 1/3 es exacta (error ~0) para polinomios de grado <= 3.
    x = np.linspace(0, 10, 101)  # 100 subintervalos (par)
    y = x ** 3
    exacta = 10.0 ** 4 / 4
    assert simpson_1_3(x, y) == pytest.approx(exacta, rel=1e-9)


def test_simpson_requiere_numero_par_de_subintervalos():
    x = np.linspace(0, 10, 10)  # 9 subintervalos (impar) -> inválido
    y = x ** 2
    with pytest.raises(ValueError):
        simpson_1_3(x, y)


def test_simpson_requiere_puntos_equiespaciados():
    x = np.array([0, 1, 2, 4, 5])  # espaciado irregular, 4 subintervalos
    y = x ** 2
    with pytest.raises(ValueError):
        simpson_1_3(x, y)


def test_simpson_es_mas_preciso_que_trapecio_en_funcion_no_polinomica():
    x = np.linspace(0, np.pi, 21)
    y = np.sin(x)
    exacta = 2.0
    error_trap = abs(trapecio(x, y) - exacta)
    error_simp = abs(simpson_1_3(x, y) - exacta)
    assert error_simp < error_trap


def test_calcular_auc_estructura_y_columnas():
    datos = generar_datos_simulacion(n=51)  # 50 subintervalos (par)
    resultado = calcular_auc(datos)
    assert list(resultado["variable"]) == ["T", "L", "E"]
    assert set(["variable", "unidad", "auc_trapecio", "auc_simpson", "diferencia_%"]) <= set(resultado.columns)
    assert (resultado["auc_trapecio"] >= 0).all()
    assert (resultado["auc_simpson"] >= 0).all()


def test_calcular_auc_diferencia_porcentual_es_pequena():
    # Con suficientes puntos, Trapecio y Simpson deben coincidir de cerca.
    datos = generar_datos_simulacion(n=101)
    resultado = calcular_auc(datos)
    assert (resultado["diferencia_%"] < 1.0).all()


def test_validar_precision_simpson_tiene_error_casi_nulo_en_cubica():
    resultado = validar_precision()
    fila = resultado.loc[resultado["caso"] == "cubica_x^3_en_[0,10]"].iloc[0]
    assert fila["error_simpson_%"] < 1e-6


def test_validar_precision_simpson_supera_a_trapecio_en_seno():
    resultado = validar_precision()
    fila = resultado.loc[resultado["caso"] == "seno_en_[0,pi]"].iloc[0]
    assert fila["error_simpson_%"] < fila["error_trapecio_%"]