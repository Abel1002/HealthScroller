"""Agente de conversación: recibe al usuario, da la bienvenida y guía.

QUÉ HACE:
    Crea el agente "conversacion": un LLM SIN herramientas que solo conversa,
    comparte datos del briefing y orienta al usuario hacia el especialista
    adecuado (estadístico, algebraico o de simulación).

PARA QUÉ SIRVE:
    Es la puerta de entrada amable del chat: responde saludos, explica qué
    hace cada agente y evita que preguntas de charla vayan a los especialistas.

CUÁNDO SE EJECUTA:
    1) Al arrancar el chatbot (graph.py lo construye con el resto del equipo).
    2) En cada turno que el orquestador etiqueta como "conversacion"
       (o cuando el alumno fuerza /agente conversacion).
"""

from ..knowledge import briefing_texto  # inyecta el resumen del notebook en el prompt (.. = un nivel arriba)
from .executor import AgenteHerramientas  # clase base que da soporte de tools (aquí, sin ninguna)

# Prompt del agente. Todo el texto va DENTRO del string (nunca comentar dentro:
# los comentarios se enviarían al LLM). Estructura: 1) identidad (HealthBot),
# 2) contexto del proyecto, 3) briefing JSON, 4) estilo y reglas de derivación.
PROMPT_CONVERSACION = f"""Eres HealthBot, el asistente conversacional del proyecto educativo HealthScroller.

El proyecto estudia cómo el uso de redes sociales afecta al sueño, la salud
mental y el rendimiento académico de 4500 estudiantes.

Datos clave del análisis (briefing del notebook):
{briefing_texto()}

Tu estilo: cálido, breve y en español. Puedes compartir los datos del briefing.
Si el usuario hace preguntas que necesitan cálculos (medias, correlaciones,
simulaciones), explícale que el orquestador puede derivarlo al agente adecuado,
o sugiérele forzarlo con /agente estadistico, /agente algebraico o /agente simulacion."""  # la f-string sustituye {briefing_texto()} por el JSON completo


def crear_agente_conversacion():
    """Devuelve el agente de conversación (sin herramientas: solo conversa)."""
    return AgenteHerramientas(  # fábrica: mismo tipo de agente que los especialistas...
        prompt=PROMPT_CONVERSACION,  # ...con su prompt...
        tools=[],  # ...pero con lista VACÍA de tools: nunca pedirá cálculos a Python
    )
