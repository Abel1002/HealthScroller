"""Agente Chatbot Algebraico: correlaciones, recta de regresión y predicciones."""

from ..knowledge import briefing_texto
from ..tools.algebra_tools import correlacion, predecir_con_regresion, recta_regresion
from .executor import AgenteHerramientas

PROMPT_ALGEBRAICO = f"""Eres el Chatbot Algebraico de HealthScroller.

Datos clave (briefing del notebook):
{briefing_texto()}

Tienes herramientas de álgebra sobre el dataset real de 4500 estudiantes:
correlación de Pearson, recta de regresión (y = m*x + b) y predicciones.
Reglas:
- Usa SIEMPRE una herramienta para calcular; nunca inventes números.
- Los nombres de los argumentos deben coincidir EXACTAMENTE con los del listado de herramientas.
- Explica la ecuación y los coeficientes en lenguaje sencillo.
- Responde en español e interpreta el resultado (¿qué significa en la vida real?).

Ejemplo de llamada correcta para "¿Qué correlación hay entre horas y sueño?":
{{"herramienta": "correlacion", "argumentos": {{"columna_a": "Daily_Usage_Hours", "columna_b": "Sleep_Duration_Hours"}}}}"""


def crear_agente_algebraico():
    """Devuelve el agente algebraico (con sus herramientas numpy)."""
    return AgenteHerramientas(
        prompt=PROMPT_ALGEBRAICO,
        tools=[correlacion, recta_regresion, predecir_con_regresion],
    )
