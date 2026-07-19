"""Pruebas del Módulo de Calibración (TG3).

Verifican que el ajuste de curvas recupera los parámetros base del
modelo dentro de una tolerancia razonable, a partir de observaciones
sintéticas con ruido controlado.
"""

from src.calibration import (
    calibrar_latencia,
    calibrar_throughput,
    generar_observaciones_ruidosas,
)
from src.simulation import (
    A_DEFAULT,
    B_DEFAULT,
    K_DEFAULT,
    T_MAX_DEFAULT,
    X0_DEFAULT,
    generar_dominio_usuarios,
    latencia,
    throughput,
)

TOLERANCIA_RELATIVA = 0.15  # 15%: margen aceptable dado el ruido introducido


def test_calibracion_throughput_recupera_parametros_base():
    x = generar_dominio_usuarios(n=80)
    t_obs = generar_observaciones_ruidosas(throughput(x, T_MAX_DEFAULT, K_DEFAULT, X0_DEFAULT))

    resultado = calibrar_throughput(x, t_obs)

    assert abs(resultado.parametros["t_max"] - T_MAX_DEFAULT) / T_MAX_DEFAULT < TOLERANCIA_RELATIVA
    assert abs(resultado.parametros["x0"] - X0_DEFAULT) / X0_DEFAULT < TOLERANCIA_RELATIVA


def test_calibracion_throughput_tiene_buen_ajuste():
    x = generar_dominio_usuarios(n=80)
    t_obs = generar_observaciones_ruidosas(throughput(x, T_MAX_DEFAULT, K_DEFAULT, X0_DEFAULT))

    resultado = calibrar_throughput(x, t_obs)

    assert resultado.r_cuadrado > 0.90


def test_calibracion_latencia_recupera_parametros_base():
    x = generar_dominio_usuarios(n=80)
    l_obs = generar_observaciones_ruidosas(latencia(x, A_DEFAULT, B_DEFAULT))

    resultado = calibrar_latencia(x, l_obs)

    assert abs(resultado.parametros["a"] - A_DEFAULT) / A_DEFAULT < TOLERANCIA_RELATIVA
    assert abs(resultado.parametros["b"] - B_DEFAULT) / B_DEFAULT < TOLERANCIA_RELATIVA


def test_calibracion_latencia_tiene_buen_ajuste():
    x = generar_dominio_usuarios(n=80)
    l_obs = generar_observaciones_ruidosas(latencia(x, A_DEFAULT, B_DEFAULT))

    resultado = calibrar_latencia(x, l_obs)

    assert resultado.r_cuadrado > 0.90
