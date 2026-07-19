"""Módulo de Calibración del Modelo Matemático.

Valida la metodología de ajuste de curvas usada para determinar los
parámetros T_max, k, x0 (throughput) y a, b (latencia) del modelo
definido en src/simulation.py.

El sistema no opera sobre un servidor real, por lo que no existe
telemetría externa contra la cual calibrar. En su lugar, se generan
observaciones sintéticas con ruido gaussiano controlado a partir de los
parámetros base del modelo, y se comprueba que el ajuste no lineal por
mínimos cuadrados (scipy.optimize.curve_fit) recupera esos parámetros
dentro de una tolerancia razonable. Esto certifica que la metodología
de ajuste sería válida si en el futuro se dispusiera de telemetría real
de un servidor bajo prueba.

Corresponde a la Tarea Global TG3 y a la sección 6.4 del informe de
Fase I.

Responsable: Valery Salas Vargas
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, para poder guardar figuras desde consola
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

RUIDO_RELATIVO = 0.05    # 5% de ruido gaussiano relativo: simula la variabilidad típica de telemetría real
SEMILLA_ALEATORIA = 42   # fija la aleatoriedad para que los resultados sean reproducibles


@dataclass
class ResultadoCalibracion:
    """Resultado de un ajuste de curvas: parámetros estimados y bondad de ajuste."""

    parametros: dict[str, float]
    r_cuadrado: float


def generar_observaciones_ruidosas(
    y_verdadero: np.ndarray,
    ruido_relativo: float = RUIDO_RELATIVO,
    semilla: int = SEMILLA_ALEATORIA,
) -> np.ndarray:
    """Simula observaciones de telemetría con ruido gaussiano relativo.

    Args:
        y_verdadero: valores "reales" de la métrica, sin ruido.
        ruido_relativo: desviación estándar del ruido, como fracción del
            valor de cada punto (ej. 0.05 equivale a 5%).
        semilla: semilla del generador aleatorio, para reproducibilidad.

    Returns:
        Arreglo del mismo tamaño de y_verdadero, con ruido agregado.
    """
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(loc=0.0, scale=ruido_relativo * np.abs(y_verdadero))
    return y_verdadero + ruido


def r_cuadrado(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el coeficiente de determinación R² de un ajuste."""
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1 - ss_res / ss_tot


def calibrar_throughput(x: np.ndarray, t_obs: np.ndarray) -> ResultadoCalibracion:
    """Ajusta la función logística de T(x) a observaciones con ruido.

    Args:
        x: usuarios concurrentes.
        t_obs: throughput observado (con ruido) para cada x.

    Returns:
        ResultadoCalibracion con T_max, k y x0 estimados, y su R².
    """
    p0 = (400.0, 0.02, 200.0)          # estimación inicial razonable para el solver
    limites = ([0, 0, 0], [2000, 1, 1000])
    (t_max, k, x0), _ = curve_fit(throughput, x, t_obs, p0=p0, bounds=limites, maxfev=10000)
    ajuste = throughput(x, t_max, k, x0)
    return ResultadoCalibracion(
        parametros={"t_max": t_max, "k": k, "x0": x0},
        r_cuadrado=r_cuadrado(t_obs, ajuste),
    )


def calibrar_latencia(x: np.ndarray, l_obs: np.ndarray) -> ResultadoCalibracion:
    """Ajusta la función exponencial de L(x) a observaciones con ruido.

    Args:
        x: usuarios concurrentes.
        l_obs: latencia observada (con ruido) para cada x.

    Returns:
        ResultadoCalibracion con a y b estimados, y su R².
    """
    p0 = (10.0, 0.01)
    limites = ([0, 0], [1000, 1])
    (a, b), _ = curve_fit(latencia, x, l_obs, p0=p0, bounds=limites, maxfev=10000)
    ajuste = latencia(x, a, b)
    return ResultadoCalibracion(
        parametros={"a": a, "b": b},
        r_cuadrado=r_cuadrado(l_obs, ajuste),
    )


def graficar_calibracion(
    x: np.ndarray,
    y_obs: np.ndarray,
    y_ajustado: np.ndarray,
    titulo: str,
    ylabel: str,
    ruta_salida: str,
) -> None:
    """Genera y guarda una gráfica de diagnóstico: datos observados vs. ajuste.

    Es una gráfica interna de validación para TG3, distinta de las
    gráficas finales del dashboard (Módulo de Visualización, TG5).

    Args:
        x: usuarios concurrentes.
        y_obs: observaciones sintéticas con ruido.
        y_ajustado: curva evaluada con los parámetros calibrados.
        titulo: título de la gráfica.
        ylabel: etiqueta del eje Y (incluir unidad).
        ruta_salida: ruta del archivo PNG a generar.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y_obs, s=14, alpha=0.6, label="Observaciones sintéticas (con ruido)")
    ax.plot(x, y_ajustado, color="crimson", linewidth=2, label="Curva calibrada")
    ax.set_title(titulo)
    ax.set_xlabel("Usuarios concurrentes (x)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def ejecutar_calibracion(n: int = 80, guardar_graficas: bool = True) -> tuple[ResultadoCalibracion, ResultadoCalibracion]:
    """Corre el flujo completo de calibración y muestra un reporte en consola.

    Args:
        n: número de puntos de muestreo del dominio de usuarios.
        guardar_graficas: si True, guarda las gráficas de diagnóstico en assets/.

    Returns:
        Tupla (resultado_throughput, resultado_latencia).
    """
    x = generar_dominio_usuarios(n=n)

    t_verdadero = throughput(x, T_MAX_DEFAULT, K_DEFAULT, X0_DEFAULT)
    l_verdadero = latencia(x, A_DEFAULT, B_DEFAULT)

    t_obs = generar_observaciones_ruidosas(t_verdadero)
    l_obs = generar_observaciones_ruidosas(l_verdadero)

    resultado_t = calibrar_throughput(x, t_obs)
    resultado_l = calibrar_latencia(x, l_obs)

    print("=== Calibración de T(x) — throughput ===")
    print(f"  Valores base (simulation.py): T_max={T_MAX_DEFAULT}, k={K_DEFAULT}, x0={X0_DEFAULT}")
    print(f"  Valores ajustados (curve_fit): "
          f"T_max={resultado_t.parametros['t_max']:.2f}, "
          f"k={resultado_t.parametros['k']:.5f}, "
          f"x0={resultado_t.parametros['x0']:.2f}")
    print(f"  R² del ajuste: {resultado_t.r_cuadrado:.4f}")
    print()
    print("=== Calibración de L(x) — latencia ===")
    print(f"  Valores base (simulation.py): a={A_DEFAULT}, b={B_DEFAULT}")
    print(f"  Valores ajustados (curve_fit): "
          f"a={resultado_l.parametros['a']:.3f}, "
          f"b={resultado_l.parametros['b']:.5f}")
    print(f"  R² del ajuste: {resultado_l.r_cuadrado:.4f}")

    if guardar_graficas:
        ajuste_t = throughput(x, **resultado_t.parametros)
        ajuste_l = latencia(x, **resultado_l.parametros)
        graficar_calibracion(
            x, t_obs, ajuste_t,
            "Calibración de T(x) — Throughput",
            "Throughput (sol/seg)",
            "assets/calibracion_throughput.png",
        )
        graficar_calibracion(
            x, l_obs, ajuste_l,
            "Calibración de L(x) — Latencia",
            "Latencia (ms)",
            "assets/calibracion_latencia.png",
        )
        print("\nGráficas guardadas en assets/calibracion_throughput.png y assets/calibracion_latencia.png")

    return resultado_t, resultado_l


if __name__ == "__main__":
    ejecutar_calibracion()
