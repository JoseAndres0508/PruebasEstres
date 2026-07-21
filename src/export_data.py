"""Módulo de Exportación de Datos.

Exporta a formato CSV los datos generados por la simulación de carga
(TG2) y los resultados del cálculo de AUC (TG4), y valida que los
archivos exportados sean íntegros (los valores leídos de vuelta
coinciden con los originales) y reproducibles (exportar los mismos
datos dos veces produce archivos idénticos, byte a byte).

Corresponde a la Tarea Global TG6.

Responsable: Jose Andrés Ortiz Marín
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def exportar_datos_simulacion(datos: pd.DataFrame, ruta: str = "data/datos_simulacion.csv") -> str:
    """Exporta el DataFrame de la simulación de carga (x, T, L, E) a CSV.

    Args:
        datos: DataFrame producido por
            src.simulation.generar_datos_simulacion().
        ruta: ruta del archivo CSV de salida. Se crean las carpetas
            intermedias si no existen.

    Returns:
        La ruta del archivo generado.
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(ruta, index=False, encoding="utf-8")
    return ruta


def exportar_resultados_auc(resultados_auc: pd.DataFrame, ruta: str = "data/resultados_auc.csv") -> str:
    """Exporta la tabla comparativa de AUC (Trapecio vs. Simpson 1/3) a CSV.

    Args:
        resultados_auc: DataFrame producido por
            src.integration.calcular_auc().
        ruta: ruta del archivo CSV de salida.

    Returns:
        La ruta del archivo generado.
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    resultados_auc.to_csv(ruta, index=False, encoding="utf-8")
    return ruta


def calcular_hash_archivo(ruta: str) -> str:
    """Calcula el hash SHA-256 del contenido de un archivo.

    Se usa para verificar reproducibilidad: dos exportaciones de los
    mismos datos deben producir archivos con el mismo hash.

    Args:
        ruta: ruta del archivo.

    Returns:
        El hash SHA-256 en hexadecimal.
    """
    with open(ruta, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def validar_integridad_exportacion(df_original: pd.DataFrame, ruta: str) -> bool:
    """Valida que el CSV exportado sea idéntico al DataFrame original.

    Relee el CSV y compara columna por columna: los nombres deben
    coincidir; las columnas numéricas se comparan con tolerancia
    flotante muy estricta (1e-9 relativa) y las columnas de texto,
    con igualdad exacta.

    Args:
        df_original: DataFrame antes de exportar.
        ruta: ruta del CSV exportado.

    Returns:
        True si el archivo es íntegro.

    Raises:
        AssertionError: si las columnas o los valores no coinciden.
    """
    df_leido = pd.read_csv(ruta)
    assert list(df_leido.columns) == list(df_original.columns), (
        f"Las columnas no coinciden: {list(df_leido.columns)} != {list(df_original.columns)}"
    )
    for columna in df_original.columns:
        original = df_original[columna]
        leido = df_leido[columna]
        if pd.api.types.is_numeric_dtype(original):
            if not np.allclose(original.to_numpy(dtype=float), leido.to_numpy(dtype=float), rtol=1e-9, atol=1e-12):
                raise AssertionError(f"Los valores numéricos de la columna '{columna}' no coinciden tras exportar/leer.")
        else:
            if not (original.astype(str).to_numpy() == leido.astype(str).to_numpy()).all():
                raise AssertionError(f"Los valores de texto de la columna '{columna}' no coinciden tras exportar/leer.")
    return True


def validar_reproducibilidad_exportacion(
    datos: pd.DataFrame,
    exportar_fn: Callable[[pd.DataFrame, str], str],
    ruta: str,
) -> bool:
    """Valida que exportar los mismos datos dos veces produzca archivos idénticos.

    Args:
        datos: DataFrame a exportar.
        exportar_fn: función de exportación a usar (exportar_datos_simulacion
            o exportar_resultados_auc).
        ruta: ruta del archivo CSV.

    Returns:
        True si ambas exportaciones producen el mismo hash SHA-256.
    """
    exportar_fn(datos, ruta)
    hash_1 = calcular_hash_archivo(ruta)
    exportar_fn(datos, ruta)
    hash_2 = calcular_hash_archivo(ruta)
    return hash_1 == hash_2


if __name__ == "__main__":
    from src.integration import calcular_auc
    from src.simulation import generar_datos_simulacion

    datos = generar_datos_simulacion(n=101)  # n impar: requisito de Simpson 1/3 (TG4)
    resultados_auc = calcular_auc(datos)

    ruta_datos = exportar_datos_simulacion(datos)
    ruta_auc = exportar_resultados_auc(resultados_auc)

    print(f"Datos de simulación exportados a: {ruta_datos}")
    print(f"  Filas: {len(datos)} | Hash SHA-256: {calcular_hash_archivo(ruta_datos)}")
    print(f"Resultados de AUC exportados a: {ruta_auc}")
    print(f"  Filas: {len(resultados_auc)} | Hash SHA-256: {calcular_hash_archivo(ruta_auc)}")

    print("\n=== Validación de integridad ===")
    integro_datos = validar_integridad_exportacion(datos, ruta_datos)
    integro_auc = validar_integridad_exportacion(resultados_auc, ruta_auc)
    print(f"  datos_simulacion.csv íntegro: {integro_datos}")
    print(f"  resultados_auc.csv íntegro: {integro_auc}")

    print("\n=== Validación de reproducibilidad ===")
    reproducible_datos = validar_reproducibilidad_exportacion(datos, exportar_datos_simulacion, ruta_datos)
    reproducible_auc = validar_reproducibilidad_exportacion(resultados_auc, exportar_resultados_auc, ruta_auc)
    print(f"  datos_simulacion.csv reproducible: {reproducible_datos}")
    print(f"  resultados_auc.csv reproducible: {reproducible_auc}")