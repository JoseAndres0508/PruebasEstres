"""Módulo de Simulación de Carga.

Genera los datos sintéticos del comportamiento de un servidor web bajo
niveles crecientes de concurrencia de usuarios, modelando las tres
métricas de rendimiento definidas en la Fase I: throughput T(x),
latencia L(x) y tasa de error E(x), en función del número de usuarios
concurrentes x.

Corresponde a la Tarea Global TG2 y a la sección 6 (Propuesta Inicial
del Modelo) del informe de Fase I.

Responsable: Wilberth Guillermo Guillén Aguilar
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Parámetros provisionales del modelo -----------------------------
# Se calibrarán con datos reales durante la Tarea Global TG3 (ajuste de
# curvas, a cargo de Valery). Estos valores por defecto solo garantizan
# que el sistema sea funcional desde ya.

T_MAX_DEFAULT = 480.0   # capacidad máxima de throughput (sol/seg)
K_DEFAULT = 0.03        # pendiente de la curva logística de T(x)
X0_DEFAULT = 250.0      # punto de inflexión de T(x) (usuarios)

A_DEFAULT = 8.0         # latencia base (ms) para L(x)
B_DEFAULT = 0.012       # tasa de crecimiento exponencial de L(x)

E_MAX_DEFAULT = 100.0   # tasa de error máxima (%) para E(x)
X_CRIT_DEFAULT = 420.0  # punto de quiebre donde el error se dispara
S_DEFAULT = 0.05        # pendiente del "escalón" de E(x)


def throughput(
    x: np.ndarray,
    t_max: float = T_MAX_DEFAULT,
    k: float = K_DEFAULT,
    x0: float = X0_DEFAULT,
) -> np.ndarray:
    """Calcula el throughput T(x) mediante una función logística.

    Fórmula (sección 6.4 del informe de Fase I):
        T(x) = T_max / (1 + e^(-k * (x - x0)))

    Args:
        x: arreglo con los valores de usuarios concurrentes.
        t_max: capacidad máxima de procesamiento del servidor (sol/seg).
        k: parámetro que controla la pendiente de la curva.
        x0: punto de inflexión donde el crecimiento se desacelera.

    Returns:
        Arreglo con el throughput T(x) para cada valor de x, en sol/seg.
    """
    return t_max / (1 + np.exp(-k * (x - x0)))


def latencia(
    x: np.ndarray,
    a: float = A_DEFAULT,
    b: float = B_DEFAULT,
) -> np.ndarray:
    """Calcula la latencia L(x) mediante una función exponencial.

    Fórmula (sección 6.4 del informe de Fase I):
        L(x) = a * e^(b * x)

    Args:
        x: arreglo con los valores de usuarios concurrentes.
        a: latencia base cuando la carga es mínima (ms).
        b: tasa de crecimiento exponencial de la latencia.

    Returns:
        Arreglo con la latencia L(x) para cada valor de x, en milisegundos.
    """
    return a * np.exp(b * x)


def tasa_error(
    x: np.ndarray,
    e_max: float = E_MAX_DEFAULT,
    x_crit: float = X_CRIT_DEFAULT,
    s: float = S_DEFAULT,
) -> np.ndarray:
    """Calcula la tasa de error E(x) mediante una función escalonada.

    Se modela como una logística de pendiente pronunciada, que se
    comporta como un escalón suavizado: permanece cercana a 0 antes del
    punto de quiebre x_crit y aumenta abruptamente después de él (ver
    sección 6.3 del informe de Fase I).

        E(x) = E_max / (1 + e^(-s * (x - x_crit)))

    Args:
        x: arreglo con los valores de usuarios concurrentes.
        e_max: tasa de error máxima que puede alcanzar el sistema (%).
        x_crit: punto de quiebre donde el sistema empieza a fallar.
        s: pendiente del escalón (mayor valor = transición más abrupta).

    Returns:
        Arreglo con la tasa de error E(x) para cada valor de x, en
        porcentaje (%).
    """
    return e_max / (1 + np.exp(-s * (x - x_crit)))


def generar_dominio_usuarios(
    x_min: int = 10,
    x_max: int = 500,
    n: int = 50,
) -> np.ndarray:
    """Genera el dominio de usuarios concurrentes (ramp-up).

    Args:
        x_min: cantidad mínima de usuarios concurrentes (carga normal).
        x_max: cantidad máxima de usuarios concurrentes (estrés extremo).
        n: número de puntos de muestreo (granularidad del dominio).

    Returns:
        Arreglo de n valores de x distribuidos uniformemente entre
        x_min y x_max.

    Raises:
        ValueError: si los límites o n no son válidos.
    """
    if x_min < 0 or x_max <= x_min:
        raise ValueError("x_max debe ser mayor que x_min y ambos no negativos.")
    if n < 2:
        raise ValueError("n debe ser al menos 2 para definir un dominio.")
    return np.linspace(x_min, x_max, n)


def generar_datos_simulacion(
    x_min: int = 10,
    x_max: int = 500,
    n: int = 50,
    t_max: float = T_MAX_DEFAULT,
    k: float = K_DEFAULT,
    x0: float = X0_DEFAULT,
    a: float = A_DEFAULT,
    b: float = B_DEFAULT,
    e_max: float = E_MAX_DEFAULT,
    x_crit: float = X_CRIT_DEFAULT,
    s: float = S_DEFAULT,
) -> pd.DataFrame:
    """Genera el conjunto de datos completo de la simulación de carga.

    Calcula T(x), L(x) y E(x) para el dominio de usuarios concurrentes
    y organiza el resultado en tuplas (x, T(x), L(x), E(x)), tal como
    se describe en la sección 6.2 del informe de Fase I.

    Args:
        x_min, x_max, n: definen el dominio de usuarios concurrentes
            (ver generar_dominio_usuarios).
        t_max, k, x0: parámetros del modelo de throughput T(x).
        a, b: parámetros del modelo de latencia L(x).
        e_max, x_crit, s: parámetros del modelo de tasa de error E(x).

    Returns:
        DataFrame con las columnas ["x", "T", "L", "E"], una fila por
        cada punto de muestreo del dominio de usuarios concurrentes.
        Esta misma estructura es la que consumirán el Módulo de Cálculo
        Integral (TG4) y el Módulo de Exportación de Datos (TG6).
    """
    x = generar_dominio_usuarios(x_min, x_max, n)
    return pd.DataFrame({
        "x": x,
        "T": throughput(x, t_max, k, x0),
        "L": latencia(x, a, b),
        "E": tasa_error(x, e_max, x_crit, s),
    })


if __name__ == "__main__":
    # Ejecución manual rápida del módulo. Pruebas formales en
    # tests/test_simulation.py.
    datos = generar_datos_simulacion()
    print(datos.head())
    print("...")
    print(datos.tail())