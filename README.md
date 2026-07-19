# Pruebas de Estrés y Análisis de Carga mediante Cálculo Integral

Proyecto de Investigación Aplicada — Fase II
Curso ISW-523 Cálculo Integral para Informática
Universidad Técnica Nacional · Carrera de Ingeniería del Software
Profesor: Maynor Rojas Bolaños

## Equipo

- Valery Salas Vargas
- Wilberth Guillermo Guillén Aguilar
- Jose Andrés Ortiz Marín

## Sobre este repositorio

Este repositorio contiene el desarrollo completo de la Fase II del
proyecto: el código fuente del sistema computacional, los datos
generados por las simulaciones, el informe final y los recursos usados
en el video demostrativo. El trabajo se organiza en torno a cuatro
módulos independientes (simulación de carga, cálculo integral numérico,
visualización y exportación de datos) que en conjunto conforman un
simulador de pruebas de carga y estrés para un servidor web.

El historial de commits documenta el avance del proyecto siguiendo el
cronograma acordado por el equipo (ver `docs/plan_de_trabajo.md`), desde
la configuración inicial del entorno hasta la entrega final el 21 de
agosto de 2026. Cada cambio relevante se registra como un commit
independiente, con mensajes que siguen la convención de Conventional
Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`), para
que sea posible rastrear qué se hizo, cuándo y por qué.

## Sobre el proyecto

Las pruebas de carga y de estrés son prácticas habituales en Ingeniería
del Software para verificar cómo se comporta un sistema bajo demanda
elevada. Sin embargo, el análisis tradicional de estas pruebas suele
limitarse a promedios estadísticos (throughput o latencia promedio), lo
que oculta picos de saturación momentáneos y ráfagas de tráfico que son,
precisamente, los momentos más críticos para entender la resiliencia de
un sistema.

Este proyecto propone una alternativa: en lugar de evaluar el
comportamiento del servidor mediante promedios aislados, se modelan las
métricas de rendimiento como funciones continuas del número de usuarios
concurrentes y se calcula el Área Bajo la Curva (AUC) mediante métodos
de integración numérica (Regla del Trapecio y Regla de Simpson 1/3).
El resultado es un indicador cuantitativo único —el esfuerzo acumulado
del servidor— que resume el comportamiento del sistema a lo largo de
toda la ventana de estrés, en vez de reducirlo a un solo número
promedio.

El sistema desarrollado en esta Fase II es un simulador que genera datos
sintéticos de carga, ajusta funciones matemáticas a esos datos y aplica
las herramientas de cálculo integral estudiadas en el curso ISW-523
para interpretar el nivel de degradación del servidor simulado.

### Variables del modelo

| Variable | Nombre               | Unidad        | Tipo de función |
|----------|-----------------------|---------------|------------------|
| `x`      | Usuarios concurrentes  | usuarios (u)  | Variable independiente |
| `T(x)`   | Throughput             | sol/seg       | Logística |
| `L(x)`   | Latencia               | ms            | Exponencial |
| `E(x)`   | Tasa de error          | %             | Escalonada |

## Estructura del repositorio

```
.
├── src/      # Código fuente de los 4 módulos del sistema
├── data/     # Datos generados por la simulación (CSV)
├── docs/     # Informe final y documentación técnica
├── assets/   # Diagramas, capturas de pantalla, recursos del video
├── tests/    # Pruebas del sistema
└── README.md
```

## Módulos del sistema

1. **Módulo de Simulación de Carga** — genera `T(x)`, `L(x)`, `E(x)` para
   un rango de usuarios concurrentes.
2. **Módulo de Cálculo Integral Numérico** — calcula el AUC mediante la
   Regla del Trapecio y la Regla de Simpson 1/3 (implementadas desde
   cero, sin funciones de librería).
3. **Módulo de Visualización** — grafica las curvas y sombrea el área
   bajo la curva; genera la tabla comparativa entre ambos métodos.
4. **Módulo de Exportación de Datos** — exporta los datos y resultados a
   formato CSV.

## Requisitos

- Python 3.10 o superior
- NumPy
- Pandas
- Matplotlib

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
python src/main.py
```

### Convenciones de código

| Elemento | Convención               | Ejemplo        |
|----------|-----------------------|---------------|
| Variables y funciones      | snake_case  | calcular_area_trapecio()  |
| Clases   | PascalCase             | SimuladorCarga       |
| Constantes   | UPPER_SNAKE_CASE               | T_MAX = 500            |
| Archivos   | snake_case, uno por módulo          | simulation.py, integration.py            |
| Type hints   | Obligatorios en la firma          | def f(x: float) -> float:             |
| Longitud de línea   | Máx. 99 caracteres          | -             |


*(Pendiente: este punto de entrada se implementará durante el desarrollo
de la Fase II — ver `docs/plan_de_trabajo.md`.)*

## Estado del proyecto

En desarrollo — Fase II. Fecha de entrega: 21 de agosto de 2026.

## Licencia

Proyecto académico desarrollado para el curso ISW-523, Universidad
Técnica Nacional. Uso educativo.
