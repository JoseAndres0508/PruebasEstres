"""Pruebas del Dashboard (TG7).

Se dividen en dos grupos:
- TestValidarParametros: prueba validar_y_normalizar_parametros(),
  función pura sin dependencias de Tkinter; corre en cualquier entorno.
- TestDashboardGUI: prueba la clase Dashboard con una ventana Tk real
  (en Linux sin pantalla física, correr con "xvfb-run -a pytest ...").

Importante: las pruebas de TestDashboardGUI comparten una única ventana
Tk para toda la clase (fixture de scope="class"), en vez de crear una
ventana nueva por prueba. Crear y destruir muchas instancias de Tk() en
el mismo proceso es frágil (se observó una falla real de Tcl/Tk en
Windows al hacerlo repetidamente). Compartir la ventana y reiniciar su
estado antes de cada prueba también refleja mejor el uso real: un
usuario no reinicia la aplicación entre cada simulación.
"""

import matplotlib.pyplot as plt
import pytest

from src.dashboard import PARAMETROS_CONFIGURABLES, Dashboard, validar_y_normalizar_parametros

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


# --- Pruebas de la función pura de validación (sin GUI) -----------------

class TestValidarParametros:
    def test_normales_no_ajusta_n(self):
        parametros, ajustado = validar_y_normalizar_parametros(VALORES_BASE)
        assert parametros["n"] == 101
        assert ajustado is False

    def test_ajusta_n_par_a_impar(self):
        valores = {**VALORES_BASE, "n": "100"}
        parametros, ajustado = validar_y_normalizar_parametros(valores)
        assert parametros["n"] == 101
        assert ajustado is True

    def test_rechaza_texto_no_numerico(self):
        valores = {**VALORES_BASE, "x_min": "diez"}
        with pytest.raises(ValueError):
            validar_y_normalizar_parametros(valores)

    def test_rechaza_x_max_menor_o_igual_a_x_min(self):
        valores = {**VALORES_BASE, "x_min": "500", "x_max": "100"}
        with pytest.raises(ValueError):
            validar_y_normalizar_parametros(valores)


# --- Pruebas de la ventana real (una sola ventana para toda la clase) ---

class TestDashboardGUI:
    @pytest.fixture(scope="class")
    @classmethod
    def app(cls):
        ventana = Dashboard()
        yield ventana
        ventana.destroy()

    @pytest.fixture(autouse=True)
    def _reiniciar_estado(self, app):
        """Reinicia campos, datos y widgets antes de cada prueba, para que
        las pruebas no interfieran entre sí al compartir una sola ventana."""
        for clave, _etiqueta, valor_defecto in PARAMETROS_CONFIGURABLES:
            app.campos[clave].set(valor_defecto)
        app.datos = None
        app.resultados_auc = None
        app.boton_exportar.state(["disabled"])
        for widget in app.area_graficas.winfo_children():
            widget.destroy()
        for widget in app.panel_resultados.winfo_children():
            widget.destroy()
        for canvas in app._canvases:
            plt.close(canvas.figure)
        app._canvases.clear()
        yield

    def test_dashboard_inicia_con_parametros_por_defecto(self, app):
        assert app.campos["x_min"].get() == "10"
        assert app.campos["n"].get() == "101"
        assert app.boton_exportar.instate(["disabled"])

    def test_ejecutar_simulacion_genera_datos_y_resultados(self, app):
        app._ejecutar_simulacion()
        assert app.datos is not None
        assert list(app.datos.columns) == ["x", "T", "L", "E"]
        assert list(app.resultados_auc["variable"]) == ["T", "L", "E"]

    def test_ejecutar_simulacion_habilita_boton_exportar(self, app):
        app._ejecutar_simulacion()
        assert not app.boton_exportar.instate(["disabled"])

    def test_ejecutar_simulacion_crea_las_graficas_y_la_tabla(self, app):
        app._ejecutar_simulacion()
        assert len(app._canvases) == 4  # 3 gráficas (T, L, E) + 1 tabla comparativa

    def test_ejecutar_simulacion_con_n_par_lo_ajusta_a_impar(self, app):
        app.campos["n"].set("100")
        app._ejecutar_simulacion()
        assert app.campos["n"].get() == "101"
        assert len(app.datos) == 101

    def test_ejecutar_simulacion_con_parametros_invalidos_no_rompe_la_app(self, app):
        app.campos["x_min"].set("no-es-un-numero")
        app._ejecutar_simulacion()  # no debe lanzar excepción; solo se cancela
        assert app.datos is None

    def test_exportar_sin_simulacion_previa_no_falla(self, app):
        app._exportar_csv()  # no debe lanzar excepción, solo mostrar advertencia

    def test_ejecutar_simulacion_dos_veces_seguidas_no_falla(self, app):
        # Simula el uso real: el usuario ejecuta más de una simulación
        # sin cerrar la ventana entre una y otra.
        app._ejecutar_simulacion()
        primera_longitud = len(app.datos)
        app.campos["x_max"].set("300")
        app._ejecutar_simulacion()
        assert len(app.datos) == primera_longitud  # mismo n, distinto rango
        assert app.datos["x"].max() <= 300