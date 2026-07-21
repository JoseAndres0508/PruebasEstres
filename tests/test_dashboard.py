"""Pruebas del Dashboard (TG7).

Se dividen en dos grupos:
- Pruebas de validar_y_normalizar_parametros(): función pura, sin
  dependencias de Tkinter, se ejecuta en cualquier entorno.
- Pruebas de la clase Dashboard: requieren un display real (en Linux,
  correr con "xvfb-run -a pytest ..." si no hay pantalla física).
"""

import pytest

from src.dashboard import Dashboard, validar_y_normalizar_parametros

VALORES_BASE = {
    "x_min": "10", "x_max": "500", "n": "101",
    "t_max": "480", "k": "0.03", "x0": "250", "a": "8", "b": "0.012",
}


@pytest.fixture(autouse=True)
def _sin_dialogos_bloqueantes(monkeypatch):
    """Evita que los messagebox reales bloqueen las pruebas esperando un clic."""
    monkeypatch.setattr("src.dashboard.messagebox.showinfo", lambda *a, **k: None)
    monkeypatch.setattr("src.dashboard.messagebox.showerror", lambda *a, **k: None)
    monkeypatch.setattr("src.dashboard.messagebox.showwarning", lambda *a, **k: None)


# --- Pruebas de la función pura de validación (sin GUI) -------------------

def test_validar_parametros_normales_no_ajusta_n():
    parametros, ajustado = validar_y_normalizar_parametros(VALORES_BASE)
    assert parametros["n"] == 101
    assert ajustado is False


def test_validar_parametros_ajusta_n_par_a_impar():
    valores = {**VALORES_BASE, "n": "100"}
    parametros, ajustado = validar_y_normalizar_parametros(valores)
    assert parametros["n"] == 101
    assert ajustado is True


def test_validar_parametros_rechaza_texto_no_numerico():
    valores = {**VALORES_BASE, "x_min": "diez"}
    with pytest.raises(ValueError):
        validar_y_normalizar_parametros(valores)


def test_validar_parametros_rechaza_x_max_menor_o_igual_a_x_min():
    valores = {**VALORES_BASE, "x_min": "500", "x_max": "100"}
    with pytest.raises(ValueError):
        validar_y_normalizar_parametros(valores)


# --- Pruebas de la ventana real (requieren display) -----------------------

@pytest.fixture
def app():
    ventana = Dashboard()
    yield ventana
    ventana.destroy()


def test_dashboard_inicia_con_parametros_por_defecto(app):
    assert app.campos["x_min"].get() == "10"
    assert app.campos["n"].get() == "101"
    assert app.boton_exportar.instate(["disabled"])


def test_ejecutar_simulacion_genera_datos_y_resultados(app):
    app._ejecutar_simulacion()
    assert app.datos is not None
    assert list(app.datos.columns) == ["x", "T", "L", "E"]
    assert list(app.resultados_auc["variable"]) == ["T", "L", "E"]


def test_ejecutar_simulacion_habilita_boton_exportar(app):
    app._ejecutar_simulacion()
    assert not app.boton_exportar.instate(["disabled"])


def test_ejecutar_simulacion_crea_las_graficas_y_la_tabla(app):
    app._ejecutar_simulacion()
    assert len(app._canvases) == 4  # 3 gráficas (T, L, E) + 1 tabla comparativa


def test_ejecutar_simulacion_con_n_par_lo_ajusta_a_impar(app):
    app.campos["n"].set("100")
    app._ejecutar_simulacion()
    assert app.campos["n"].get() == "101"
    assert len(app.datos) == 101


def test_ejecutar_simulacion_con_parametros_invalidos_no_rompe_la_app(app):
    app.campos["x_min"].set("no-es-un-numero")
    app._ejecutar_simulacion()  # no debe lanzar excepción; solo se cancela
    assert app.datos is None


def test_exportar_sin_simulacion_previa_no_falla(app):
    app._exportar_csv()  # no debe lanzar excepción, solo mostrar advertencia