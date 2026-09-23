"""Agente Chatbot de Simulación: Monte Carlo, bootstrap y escenarios what-if.

QUÉ HACE:
    Crea el agente "simulacion": un LLM con 3 herramientas de muestreo
    (monte_carlo_media, bootstrap_intervalo_confianza, escenario_what_if)
    para responder preguntas de probabilidad y escenarios contrafácticos.

PARA QUÉ SIRVE:
    Especialista al que el orquestador deriva preguntas tipo "¿qué pasaría si
    reduzco el uso a 3 horas?" o "¿cuál es el intervalo de confianza...?".

CUÁNDO SE EJECUTA:
    1) Al arrancar el chatbot (graph.py lo construye con el resto del equipo).
    2) En cada turno etiquetado "simulacion" (o con /agente simulacion).
    Sus tools pueden tardar unos segundos (miles de muestreos).
"""

from ..knowledge import briefing_texto  # briefing del notebook inyectado en el prompt
from ..tools.simulation_tools import (  # las 3 tools de muestreo disponibles
    bootstrap_intervalo_confianza,
    escenario_what_if,
    monte_carlo_media,
)
from .executor import AgenteHerramientas  # clase con el bucle de herramientas

# Prompt del especialista. Todo el texto va DENTRO del string (nunca comentar
# dentro: los comentarios se enviarían al LLM). Estructura: 1) identidad,
# 2) briefing, 3) tools disponibles, 4) reglas (nunca inventar + explicar qué
# es cada simulación), 5) ejemplo few-shot de JSON.
PROMPT_SIMULACION = f"""Eres el Chatbot de Simulación de HealthScroller.

Datos clave (briefing del notebook):
{briefing_texto()}

Tienes herramientas para simular sobre el dataset real de 4500 estudiantes:
Monte Carlo, bootstrap (intervalos de confianza) y escenarios "¿qué pasaría si...?".
Reglas:
- Usa SIEMPRE una herramienta; nunca inventes números.
- Los nombres de los argumentos deben coincidir EXACTAMENTE con los del listado de herramientas.
- Responde en español y explica en simple qué es la simulación elegida.
- Cierra con la interpretación práctica del resultado.

Ejemplo de llamada correcta para "¿Qué pasaría si bajo de 7 a 3 horas?":
{{"herramienta": "escenario_what_if", "argumentos": {{"columna_x": "Daily_Usage_Hours", "columna_y": "Academic_Performance_GPA", "valor_actual": 7.0, "valor_nuevo": 3.0}}}}"""  # {{ }} = llaves dobles: la f-string imprime el {literal} del JSON de ejemplo


def crear_agente_simulacion():
    """Devuelve el agente de simulación (con sus herramientas de muestreo)."""
    return AgenteHerramientas(  # fábrica genérica: prompt + lista de tools
        prompt=PROMPT_SIMULACION,  # personalidad y reglas de este especialista
        tools=[monte_carlo_media, bootstrap_intervalo_confianza, escenario_what_if],  # las 3 tools de simulation_tools.py
    )
