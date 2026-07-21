"""Dashboard — Interfaz gráfica del Simulador de Pruebas de Carga y Estrés.

Integra los cuatro módulos del sistema (simulación, cálculo integral,
visualización y exportación) en una única ventana de escritorio
(Tkinter), siguiendo el flujo de 7 etapas descrito en la sección 7.6
del informe de Fase I:

  1. Se abre la aplicación con los parámetros por defecto ya cargados
     en el panel de configuración.
  2. El usuario revisa y, si lo desea, ajusta esos parámetros.
  3. Al presionar "Ejecutar simulación" se invoca el Módulo de
     Simulación de Carga (src.simulation): calcula T(x), L(x), E(x).
  4. Los arreglos generados se transfieren al Módulo de Cálculo
     Integral Numérico (src.integration): AUC por Trapecio y Simpson 1/3.
  5. El Módulo de Visualización (src.visualization) genera las 3
     gráficas con el AUC sombreado y las despliega en el área central.
  6. El panel de resultados se actualiza con la tabla comparativa de
     AUC y se habilita el botón de exportación (src.export_data).
  7. El usuario interpreta los resultados (fuera del alcance del código:
     punto de saturación, magnitud del esfuerzo acumulado, precisión
     relativa de ambos métodos de integración).

Solo se exponen en el panel de configuración los parámetros descritos
en la sección 7.5 del informe de Fase I (x_min, x_max, n, T_max, k, x0,
a, b). Los parámetros de E(x) no se exponen porque el informe de Fase I
no define una fórmula explícita para esa variable (ver nota de diseño
en la documentación técnica, sección 2.3); se usan los valores por
defecto de src.simulation.

Corresponde a la Tarea Global TG7.

Responsable: Wilberth Guillermo Guillén Aguilar
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.export_data import exportar_datos_simulacion, exportar_resultados_auc
from src.integration import calcular_auc
from src.simulation import (
    A_DEFAULT,
    B_DEFAULT,
    K_DEFAULT,
    T_MAX_DEFAULT,
    X0_DEFAULT,
    generar_datos_simulacion,
)
from src.visualization import graficar_todas_las_variables, tabla_comparativa_auc

# Parámetros expuestos en el panel de configuración: clave, etiqueta, valor por defecto.
PARAMETROS_CONFIGURABLES = [
    ("x_min", "Usuarios concurrentes — mínimo (x_min)", "10"),
    ("x_max", "Usuarios concurrentes — máximo (x_max)", "500"),
    ("n", "Puntos de muestreo (n, debe ser impar)", "101"),
    ("t_max", "T_max — throughput máximo (sol/seg)", str(T_MAX_DEFAULT)),
    ("k", "k — pendiente de T(x)", str(K_DEFAULT)),
    ("x0", "x0 — punto de inflexión de T(x)", str(X0_DEFAULT)),
    ("a", "a — latencia base (ms)", str(A_DEFAULT)),
    ("b", "b — tasa de crecimiento de L(x)", str(B_DEFAULT)),
]


def validar_y_normalizar_parametros(valores: dict[str, str]) -> tuple[dict[str, float], bool]:
    """Convierte y valida los parámetros ingresados en el formulario.

    Función pura (sin dependencias de Tkinter), separada a propósito
    para poder probarse sin necesidad de una pantalla o display.

    Args:
        valores: diccionario {clave: texto} tal como llega de los
            campos de entrada (ver PARAMETROS_CONFIGURABLES).

    Returns:
        Tupla (parametros, n_fue_ajustado):
            parametros: diccionario con los valores ya convertidos a
                int/float, listos para generar_datos_simulacion().
            n_fue_ajustado: True si n era par y se ajustó a impar
                (requisito de la Regla de Simpson 1/3, ver TG4).

    Raises:
        ValueError: si algún valor no es numérico, o si x_max <= x_min.
    """
    try:
        x_min = int(float(valores["x_min"]))
        x_max = int(float(valores["x_max"]))
        n = int(float(valores["n"]))
        t_max = float(valores["t_max"])
        k = float(valores["k"])
        x0 = float(valores["x0"])
        a = float(valores["a"])
        b = float(valores["b"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Todos los parámetros deben ser numéricos.") from exc

    if x_max <= x_min:
        raise ValueError("x_max debe ser mayor que x_min.")

    n_fue_ajustado = False
    if n % 2 == 0:
        n += 1  # Simpson 1/3 exige número impar de puntos (ver src/integration.py)
        n_fue_ajustado = True

    parametros = {
        "x_min": x_min, "x_max": x_max, "n": n,
        "t_max": t_max, "k": k, "x0": x0, "a": a, "b": b,
    }
    return parametros, n_fue_ajustado


class Dashboard(tk.Tk):
    """Ventana principal: panel de configuración, gráficas y resultados."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Simulador de Pruebas de Estrés y Análisis de Carga — ISW-523")
        self.geometry("1150x780")

        self.datos: pd.DataFrame | None = None
        self.resultados_auc: pd.DataFrame | None = None
        self._canvases: list[FigureCanvasTkAgg] = []

        self._construir_panel_configuracion()
        self._construir_area_visualizacion()
        self._construir_panel_resultados()

    # --- Etapas 1 y 2: panel de configuración -------------------------
    def _construir_panel_configuracion(self) -> None:
        panel = ttk.LabelFrame(self, text="Configuración")
        panel.pack(side="top", fill="x", padx=10, pady=(10, 5))

        self.campos: dict[str, tk.StringVar] = {}
        for i, (clave, etiqueta, valor_defecto) in enumerate(PARAMETROS_CONFIGURABLES):
            fila, columna = divmod(i, 4)
            ttk.Label(panel, text=etiqueta).grid(row=fila * 2, column=columna, sticky="w", padx=5, pady=(5, 0))
            variable = tk.StringVar(value=valor_defecto)
            ttk.Entry(panel, textvariable=variable, width=16).grid(
                row=fila * 2 + 1, column=columna, sticky="w", padx=5, pady=(0, 5)
            )
            self.campos[clave] = variable

        self.boton_ejecutar = ttk.Button(panel, text="Ejecutar simulación", command=self._ejecutar_simulacion)
        self.boton_ejecutar.grid(row=0, column=4, rowspan=2, padx=15, pady=5, sticky="ns")

        self.boton_exportar = ttk.Button(panel, text="Exportar a CSV", command=self._exportar_csv, state="disabled")
        self.boton_exportar.grid(row=2, column=4, rowspan=2, padx=15, pady=5, sticky="ns")

    # --- Etapa 5: área de visualización --------------------------------
    def _construir_area_visualizacion(self) -> None:
        self.area_graficas = ttk.LabelFrame(self, text="Gráficas — T(x), L(x), E(x) con AUC sombreado")
        self.area_graficas.pack(side="top", fill="both", expand=True, padx=10, pady=5)

    # --- Etapa 6: panel de resultados numéricos ------------------------
    def _construir_panel_resultados(self) -> None:
        self.panel_resultados = ttk.LabelFrame(self, text="Resultados — AUC (Trapecio vs. Simpson 1/3)")
        self.panel_resultados.pack(side="bottom", fill="x", padx=10, pady=(5, 10))

    # --- Etapas 3 a 6: flujo completo al presionar "Ejecutar simulación" ---
    def _ejecutar_simulacion(self) -> None:
        valores_texto = {clave: variable.get() for clave, variable in self.campos.items()}
        try:
            parametros, n_fue_ajustado = validar_y_normalizar_parametros(valores_texto)
        except ValueError as error:
            messagebox.showerror("Parámetros inválidos", str(error))
            return

        if n_fue_ajustado:
            self.campos["n"].set(str(parametros["n"]))
            messagebox.showinfo(
                "Ajuste automático",
                f"n debe ser impar para la Regla de Simpson 1/3. Se ajustó a n={parametros['n']}.",
            )

        # Etapa 3: Módulo de Simulación de Carga
        self.datos = generar_datos_simulacion(
            x_min=parametros["x_min"], x_max=parametros["x_max"], n=parametros["n"],
            t_max=parametros["t_max"], k=parametros["k"], x0=parametros["x0"],
            a=parametros["a"], b=parametros["b"],
        )

        # Etapa 4: Módulo de Cálculo Integral Numérico
        self.resultados_auc = calcular_auc(self.datos)

        # Etapa 5: Módulo de Visualización
        self._actualizar_graficas()

        # Etapa 6: panel de resultados + habilitar exportación
        self._actualizar_panel_resultados()
        self.boton_exportar.state(["!disabled"])

    def _actualizar_graficas(self) -> None:
        for widget in self.area_graficas.winfo_children():
            widget.destroy()
        for canvas in self._canvases:
            plt.close(canvas.figure)
        self._canvases.clear()

        figuras = graficar_todas_las_variables(self.datos)
        contenedor = ttk.Frame(self.area_graficas)
        contenedor.pack(fill="both", expand=True)
        for figura in figuras.values():
            figura.set_size_inches(3.6, 2.8)
            canvas = FigureCanvasTkAgg(figura, master=contenedor)
            canvas.draw()
            canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=5)
            self._canvases.append(canvas)

    def _actualizar_panel_resultados(self) -> None:
        for widget in self.panel_resultados.winfo_children():
            widget.destroy()

        figura_tabla = tabla_comparativa_auc(self.resultados_auc)
        figura_tabla.set_size_inches(8.5, 1.6)
        canvas = FigureCanvasTkAgg(figura_tabla, master=self.panel_resultados)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x")
        self._canvases.append(canvas)

    # --- Etapa 6 (exportación) -----------------------------------------
    def _exportar_csv(self) -> None:
        if self.datos is None or self.resultados_auc is None:
            messagebox.showwarning("Nada que exportar", "Ejecuta la simulación antes de exportar.")
            return
        ruta_datos = exportar_datos_simulacion(self.datos)
        ruta_auc = exportar_resultados_auc(self.resultados_auc)
        messagebox.showinfo("Exportación completa", f"Datos exportados a:\n{ruta_datos}\n{ruta_auc}")


def iniciar_dashboard() -> None:
    """Punto de entrada: crea y corre la ventana principal del dashboard."""
    app = Dashboard()
    app.mainloop()


if __name__ == "__main__":
    iniciar_dashboard()