"""Pruebas del Módulo de Exportación de Datos (TG6)."""

import pandas as pd
import pytest

from src.export_data import (
    calcular_hash_archivo,
    exportar_datos_simulacion,
    exportar_resultados_auc,
    validar_integridad_exportacion,
    validar_reproducibilidad_exportacion,
)
from src.integration import calcular_auc
from src.simulation import generar_datos_simulacion


@pytest.fixture
def datos_muestra() -> pd.DataFrame:
    return generar_datos_simulacion(n=51)


@pytest.fixture
def auc_muestra() -> pd.DataFrame:
    return calcular_auc(generar_datos_simulacion(n=101))  # n impar, requisito de Simpson 1/3


def test_exportar_datos_simulacion_crea_archivo(tmp_path, datos_muestra):
    ruta = tmp_path / "datos.csv"
    resultado = exportar_datos_simulacion(datos_muestra, str(ruta))
    assert ruta.exists()
    assert resultado == str(ruta)


def test_exportar_resultados_auc_crea_archivo(tmp_path, auc_muestra):
    ruta = tmp_path / "auc.csv"
    exportar_resultados_auc(auc_muestra, str(ruta))
    assert ruta.exists()


def test_exportar_crea_carpetas_intermedias(tmp_path, datos_muestra):
    ruta = tmp_path / "subcarpeta" / "otra" / "datos.csv"
    exportar_datos_simulacion(datos_muestra, str(ruta))
    assert ruta.exists()


def test_validar_integridad_exportacion_datos_simulacion(tmp_path, datos_muestra):
    ruta = tmp_path / "datos.csv"
    exportar_datos_simulacion(datos_muestra, str(ruta))
    assert validar_integridad_exportacion(datos_muestra, str(ruta)) is True


def test_validar_integridad_exportacion_resultados_auc(tmp_path, auc_muestra):
    # Incluye columnas de texto (variable, unidad) y numéricas: valida ambos casos.
    ruta = tmp_path / "auc.csv"
    exportar_resultados_auc(auc_muestra, str(ruta))
    assert validar_integridad_exportacion(auc_muestra, str(ruta)) is True


def test_validar_integridad_detecta_columnas_distintas(tmp_path, datos_muestra):
    ruta = tmp_path / "datos_corruptos.csv"
    df_distinto = datos_muestra.rename(columns={"T": "Throughput"})
    df_distinto.to_csv(ruta, index=False)
    with pytest.raises(AssertionError):
        validar_integridad_exportacion(datos_muestra, str(ruta))


def test_validar_integridad_detecta_valores_alterados(tmp_path, datos_muestra):
    ruta = tmp_path / "datos_alterados.csv"
    df_alterado = datos_muestra.copy()
    df_alterado.loc[0, "T"] = df_alterado.loc[0, "T"] + 1000  # se altera un valor a propósito
    df_alterado.to_csv(ruta, index=False)
    with pytest.raises(AssertionError):
        validar_integridad_exportacion(datos_muestra, str(ruta))


def test_calcular_hash_archivo_es_determinista(tmp_path, datos_muestra):
    ruta = tmp_path / "datos.csv"
    exportar_datos_simulacion(datos_muestra, str(ruta))
    hash_1 = calcular_hash_archivo(str(ruta))
    hash_2 = calcular_hash_archivo(str(ruta))
    assert hash_1 == hash_2
    assert len(hash_1) == 64  # SHA-256 en hexadecimal


def test_validar_reproducibilidad_exportacion_datos(tmp_path, datos_muestra):
    ruta = tmp_path / "datos.csv"
    assert validar_reproducibilidad_exportacion(datos_muestra, exportar_datos_simulacion, str(ruta)) is True


def test_validar_reproducibilidad_exportacion_auc(tmp_path, auc_muestra):
    ruta = tmp_path / "auc.csv"
    assert validar_reproducibilidad_exportacion(auc_muestra, exportar_resultados_auc, str(ruta)) is True