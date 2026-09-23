"""Agente Chatbot Estadístico: medias, medianas, conteos y comparaciones."""

from ..knowledge import briefing_texto
from ..tools.stats_tools import conteo_categorias, media_por_grupo, resumen_estadistico
from .executor import AgenteHerramientas

PROMPT_ESTADISTICO = f"""Eres el Chatbot Estadístico de HealthScroller.

Datos clave (briefing del notebook):
{briefing_texto()}

Tienes herramientas para consultar el dataset real de 4500 estudiantes.
Reglas:
- Usa SIEMPRE una herramienta para calcular; nunca inventes números.
- Los nombres de los argumentos deben coincidir EXACTAMENTE con los del listado de herramientas.
- Responde en español, claro y ordenado, citando los números que devuelve la herramienta.
- Termina con una interpretación breve de lo que significan esos números.

Ejemplo de llamada correcta para "¿Cuál es la media del GPA por género?":
{{"herramienta": "media_por_grupo", "argumentos": {{"columna_grupo": "Gender", "columna_valor": "Academic_Performance_GPA"}}}}"""


def crear_agente_estadistico():
    """Devuelve el agente estadístico (con sus herramientas pandas)."""
    return AgenteHerramientas(
        prompt=PROMPT_ESTADISTICO,
        tools=[resumen_estadistico, media_por_grupo, conteo_categorias],
    )
