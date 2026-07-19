"""Módulo de Cálculo Integral Numérico.

Implementa la Regla del Trapecio compuesta y la Regla de Simpson 1/3
"desde cero" (sin delegar a np.trapz, scipy.integrate ni funciones
similares de librerías externas), tal como exige la TG4. Ambas se usan
para calcular el Área Bajo la Curva (AUC) de T(x), L(x) y E(x), que
representa el esfuerzo acumulado del servidor durante la ventana de
estrés (ver sección 5.2 del informe de Fase I).

Corresponde a la Tarea Global TG4.

Responsable: Wilberth Guillermo Guillén Aguilar
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation import generar_datos_simulacion

# Columna -> unidad del AUC resultante (dimensión recurso × usuarios,
# análoga a "recurso-tiempo" de la sección 5.2.3 del informe de Fase I,
# pero integrada respecto a usuarios concurrentes en vez de tiempo).
UNIDADES_AUC = {
    "T": "sol/seg · usuarios",
    "L": "ms · usuarios",
    "E": "% · usuarios",
}


def trapecio(x: np.ndarray, y: np.ndarray) -> float:
    """Aproxima la integral definida mediante la Regla del Trapecio compuesta.

    Implementación manual, válida para puntos equiespaciados o no:
        AUC ≈ Σ [(y_i + y_{i+1}) / 2] * (x_{i+1} - x_i)

    Corresponde a la fórmula de la sección 5.2.2 del informe de Fase I.

    Args:
        x: valores de la variable independiente (ordenados ascendentemente).
        y: valores de la función evaluada en cada x.

    Returns:
        Aproximación del área bajo la curva (AUC).

    Raises:
        ValueError: si x e y no tienen la misma longitud o hay menos de 2 puntos.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud.")
    if len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos para integrar.")

    total = 0.0
    for i in range(len(x) - 1):
        ancho = x[i + 1] - x[i]
        alto_promedio = (y[i] + y[i + 1]) / 2
        total += ancho * alto_promedio
    return total


def simpson_1_3(x: np.ndarray, y: np.ndarray) -> float:
    """Aproxima la integral definida mediante la Regla de Simpson 1/3 compuesta.

    Implementación manual:
        AUC ≈ (h/3) * [f0 + 4f1 + 2f2 + 4f3 + ... + 4f_{n-1} + fn]

    Corresponde a la fórmula de la sección 5.2.3 del informe de Fase I.
    Requiere puntos equiespaciados y un número PAR de subintervalos
    (es decir, un número IMPAR de puntos).

    Args:
        x: valores de la variable independiente, equiespaciados.
        y: valores de la función evaluada en cada x.

    Returns:
        Aproximación del área bajo la curva (AUC).

    Raises:
        ValueError: si x e y no tienen la misma longitud, si hay menos
            de 3 puntos, si el número de subintervalos es impar, o si
            los puntos no están equiespaciados.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud.")

    n = len(x) - 1  # número de subintervalos
    if n < 2:
        raise ValueError("Se requieren al menos 3 puntos para Simpson 1/3.")
    if n % 2 != 0:
        raise ValueError(
            "La Regla de Simpson 1/3 requiere un número par de subintervalos "
            "(número impar de puntos). Ajuste n en el dominio de usuarios."
        )

    pasos = np.diff(x)
    if not np.allclose(pasos, pasos[0]):
        raise ValueError("La Regla de Simpson 1/3 requiere puntos equiespaciados.")

    h = pasos[0]
    suma = y[0] + y[-1]
    for i in range(1, n):
        coeficiente = 4 if i % 2 != 0 else 2
        suma += coeficiente * y[i]
    return (h / 3) * suma


def calcular_auc(datos: pd.DataFrame) -> pd.DataFrame:
    """Calcula el AUC de T, L y E con ambos métodos y su diferencia porcentual.

    Args:
        datos: DataFrame con columnas ["x", "T", "L", "E"], tal como lo
            produce generar_datos_simulacion() (ver src/simulation.py).
            Debe tener un número impar de filas (para que Simpson 1/3
            sea aplicable).

    Returns:
        DataFrame con columnas [variable, unidad, auc_trapecio,
        auc_simpson, diferencia_%], una fila por cada variable T, L, E.
        La diferencia porcentual se calcula tomando Simpson 1/3 como
        referencia (es el método de mayor orden de precisión).
    """
    x = datos["x"].to_numpy()
    filas = []
    for columna in ("T", "L", "E"):
        y = datos[columna].to_numpy()
        auc_trap = trapecio(x, y)
        auc_simp = simpson_1_3(x, y)
        diferencia_pct = abs(auc_trap - auc_simp) / abs(auc_simp) * 100 if auc_simp != 0 else float("nan")
        filas.append({
            "variable": columna,
            "unidad": UNIDADES_AUC[columna],
            "auc_trapecio": auc_trap,
            "auc_simpson": auc_simp,
            "diferencia_%": diferencia_pct,
        })
    return pd.DataFrame(filas)


# --- Validación con integrales de solución conocida ------------------
# Simpson 1/3 es exacta (error ≈ 0) para cualquier polinomio de grado
# <= 3, por eso se usa x^3 como caso de control. sin(x) se agrega como
# segundo caso, con solución conocida pero no polinómica, para observar
# el comportamiento típico (Simpson más preciso que Trapecio) en una
# función que sí es representativa del tipo de curvas suaves de T(x)/L(x).
CASOS_VALIDACION = {
    "cubica_x^3_en_[0,10]": {
        "f": lambda x: x ** 3,
        "a": 0.0,
        "b": 10.0,
        "integral_exacta": 10.0 ** 4 / 4,  # ∫0^10 x^3 dx = 2500
    },
    "seno_en_[0,pi]": {
        "f": np.sin,
        "a": 0.0,
        "b": np.pi,
        "integral_exacta": 2.0,  # ∫0^pi sin(x) dx = 2
    },
}


def validar_precision(n: int = 101) -> pd.DataFrame:
    """Valida trapecio() y simpson_1_3() contra integrales de solución conocida.

    Args:
        n: número de puntos de muestreo (debe ser impar para Simpson 1/3).

    Returns:
        DataFrame con el error porcentual de cada método en cada caso
        de validación.
    """
    filas = []
    for nombre, caso in CASOS_VALIDACION.items():
        x = np.linspace(caso["a"], caso["b"], n)
        y = caso["f"](x)
        auc_trap = trapecio(x, y)
        auc_simp = simpson_1_3(x, y)
        exacta = caso["integral_exacta"]
        filas.append({
            "caso": nombre,
            "integral_exacta": exacta,
            "auc_trapecio": auc_trap,
            "error_trapecio_%": abs(auc_trap - exacta) / abs(exacta) * 100,
            "auc_simpson": auc_simp,
            "error_simpson_%": abs(auc_simp - exacta) / abs(exacta) * 100,
        })
    return pd.DataFrame(filas)


def ejecutar_calculo_integral(n: int = 101) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre la validación y el cálculo de AUC sobre los datos simulados.

    Args:
        n: número de puntos de muestreo del dominio de usuarios
            concurrentes (debe ser impar; ver simpson_1_3).

    Returns:
        Tupla (validacion, auc_resultados).
    """
    print("=== Validación con integrales de solución conocida ===")
    validacion = validar_precision()
    with pd.option_context("display.float_format", lambda v: f"{v:,.6f}"):
        print(validacion.to_string(index=False))

    print("\n=== AUC de T(x), L(x), E(x) — datos simulados (TG2) ===")
    datos = generar_datos_simulacion(n=n)
    auc_resultados = calcular_auc(datos)
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
        print(auc_resultados.to_string(index=False))

    return validacion, auc_resultados


if __name__ == "__main__":
    ejecutar_calculo_integral()