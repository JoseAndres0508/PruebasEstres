"""Punto de entrada del sistema.

Inicia la interfaz gráfica (dashboard) que integra los cuatro módulos
del sistema: simulación de carga, cálculo integral numérico,
visualización y exportación de datos.

Uso:
    python -m src.main
"""

from src.dashboard import iniciar_dashboard

if __name__ == "__main__":
    iniciar_dashboard()