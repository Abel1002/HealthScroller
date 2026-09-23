"""Agente Chatbot Algebraico: correlaciones, recta de regresión y predicciones.

QUÉ HACE:
    Crea el agente "algebraico": un LLM con 3 herramientas de numpy/pandas
    (correlacion, recta_regresion, predecir_con_regresion) para responder
    preguntas sobre RELACIÓN entre variables con números exactos.

PARA QUÉ SIRVE:
    Especialista al que el orquestador deriva preguntas como "¿qué
    correlación hay entre horas de uso y sueño?" o predicciones puntuales.

CUÁNDO SE EJECUTA:
    1) Al arrancar el chatbot (graph.py lo construye con el resto del equipo).
    2) En cada turno etiquetado "algebraico" (o con /agente algebraico).
"""

from ..knowledge import briefing_texto  # briefing del notebook inyectado en el prompt
from ..tools.algebra_tools import correlacion, predecir_con_regresion, recta_regresion  # las 3 tools de álgebra disponibles
from .executor import AgenteHerramientas  # clase con el bucle de herramientas

# Prompt del especialista. Todo el texto va DENTRO del string (nunca comentar
# dentro: los comentarios se enviarían al LLM). Estructura: 1) identidad,
# 2) briefing, 3) tools disponibles, 4) reglas (cifras EXACTAS y explicar la
# ecuación en simple), 5) ejemplo few-shot de JSON.
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
{{"herramienta": "correlacion", "argumentos": {{"columna_a": "Daily_Usage_Hours", "columna_b": "Sleep_Duration_Hours"}}}}"""  # {{ }} = llaves dobles: la f-string imprime el {literal} del JSON de ejemplo


def crear_agente_algebraico():
    """Devuelve el agente algebraico (con sus herramientas numpy)."""
    return AgenteHerramientas(  # fábrica genérica: prompt + lista de tools
        prompt=PROMPT_ALGEBRAICO,  # personalidad y reglas de este especialista
        tools=[correlacion, recta_regresion, predecir_con_regresion],  # las 3 tools de algebra_tools.py
    )
