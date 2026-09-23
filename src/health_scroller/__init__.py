"""HealthScroller: chatbot multi-agente (LangGraph + Ollama) sobre redes sociales y salud estudiantil.

QUÉ HACE:
    Declara que la carpeta `health_scroller/` es un paquete Python y fija su versión.

PARA QUÉ SIRVE:
    Gracias a este fichero, el resto del código puede importar módulos con rutas
    limpias (p. ej. `from health_scroller.config import OLLAMA_MODEL`).

CUÁNDO SE EJECUTA:
    Automáticamente, UNA sola vez, en el primer `import health_scroller`
    (por ejemplo al lanzar `python -m health_scroller.cli`).
"""

# Versión del paquete: identifica esta instalación (la lee pyproject.toml como metadato)
__version__ = "1.0.0"
