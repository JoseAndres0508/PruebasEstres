"""Pruebas unitarias del Módulo de Simulación de Carga (TG2)."""

import numpy as np
import pytest

from src.simulation import (
    generar_datos_simulacion,
    generar_dominio_usuarios,
    latencia,
    tasa_error,
    throughput,
)


def test_dominio_usuarios_tamano_y_limites():
    x = generar_dominio_usuarios(x_min=10, x_max=500, n=50)
    assert len(x) == 50
    assert x[0] == 10
    assert x[-1] == 500


def test_dominio_usuarios_valida_argumentos():
    with pytest.raises(ValueError):
        generar_dominio_usuarios(x_min=100, x_max=50, n=10)
    with pytest.raises(ValueError):
        generar_dominio_usuarios(n=1)


def test_throughput_no_negativo_y_acotado():
    x = generar_dominio_usuarios()
    t = throughput(x, t_max=480)
    assert np.all(t >= 0)
    assert np.all(t <= 480)


def test_throughput_es_creciente_en_fase_inicial():
    x = np.array([10, 50, 100])
    t = throughput(x)
    assert t[0] < t[1] < t[2]


def test_latencia_crece_con_x():
    x = np.array([10, 100, 400])
    l = latencia(x)
    assert l[0] < l[1] < l[2]


def test_tasa_error_cercana_a_cero_en_carga_baja():
    e = tasa_error(np.array([10]), x_crit=420)
    assert e[0] < 1.0  # % de error casi nulo lejos del punto de quiebre


def test_tasa_error_alta_tras_punto_de_quiebre():
    e = tasa_error(np.array([500]), x_crit=420, e_max=100)
    assert e[0] > 90.0  # cerca del máximo tras el punto de quiebre


def test_generar_datos_simulacion_estructura():
    datos = generar_datos_simulacion(n=20)
    assert list(datos.columns) == ["x", "T", "L", "E"]
    assert len(datos) == 20
    assert not datos.isnull().values.any()