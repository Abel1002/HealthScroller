"""Agente Chatbot de Simulación: Monte Carlo, bootstrap y escenarios what-if."""

from ..knowledge import briefing_texto
from ..tools.simulation_tools import (
    bootstrap_intervalo_confianza,
    escenario_what_if,
    monte_carlo_media,
)
from .executor import AgenteHerramientas

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
{{"herramienta": "escenario_what_if", "argumentos": {{"columna_x": "Daily_Usage_Hours", "columna_y": "Academic_Performance_GPA", "valor_actual": 7.0, "valor_nuevo": 3.0}}}}"""


def crear_agente_simulacion():
    """Devuelve el agente de simulación (con sus herramientas de muestreo)."""
    return AgenteHerramientas(
        prompt=PROMPT_SIMULACION,
        tools=[monte_carlo_media, bootstrap_intervalo_confianza, escenario_what_if],
    )
