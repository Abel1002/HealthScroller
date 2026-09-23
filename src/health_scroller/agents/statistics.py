"""Agente Chatbot Estadístico: medias, medianas, conteos y comparaciones.

QUÉ HACE:
    Crea el agente "estadistico": un LLM con 3 herramientas de pandas
    (resumen_estadistico, media_por_grupo, conteo_categorias) para responder
    preguntas de estadística descriptiva con datos EXACTOS del dataset.

PARA QUÉ SIRVE:
    Especialista al que el orquestador deriva preguntas como "¿cuál es la
    media del GPA por género?" o "¿cuántos usan TikTok?".

CUÁNDO SE EJECUTA:
    1) Al arrancar el chatbot (graph.py lo construye con el resto del equipo).
    2) En cada turno etiquetado "estadistico" (o con /agente estadistico).
"""

from ..knowledge import briefing_texto  # briefing del notebook inyectado en el prompt (contexto precalculado)
from ..tools.stats_tools import conteo_categorias, media_por_grupo, resumen_estadistico  # las 3 tools de pandas que usará
from .executor import AgenteHerramientas  # clase que implementa el bucle de herramientas

# Prompt del especialista. Todo el texto va DENTRO del string (nunca comentar
# dentro: los comentarios se enviarían al LLM). Estructura: 1) identidad,
# 2) briefing JSON, 3) qué tools tiene, 4) reglas duras (nunca inventar),
# 5) ejemplo few-shot de JSON (la pista principal de gemma3:1b).
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
{{"herramienta": "media_por_grupo", "argumentos": {{"columna_grupo": "Gender", "columna_valor": "Academic_Performance_GPA"}}}}"""  # {{ }} = llaves dobles para que la f-string imprima {literal} del JSON de ejemplo


def crear_agente_estadistico():
    """Devuelve el agente estadístico (con sus herramientas pandas)."""
    return AgenteHerramientas(  # fábrica genérica: prompt + lista de tools
        prompt=PROMPT_ESTADISTICO,  # personalidad y reglas de este especialista
        tools=[resumen_estadistico, media_por_grupo, conteo_categorias],  # las 3 tools de stats_tools.py disponibles para él
    )
