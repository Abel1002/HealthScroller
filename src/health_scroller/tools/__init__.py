"""Herramientas de los agentes: funciones pandas/numpy que el LLM puede invocar.

QUÉ HACE:
    Convierte la carpeta `tools/` en el subpaquete `health_scroller.tools`.

PARA QUÉ SIRVE:
    Permite importar las herramientas desde cualquier sitio con rutas limpias:
        from health_scroller.tools.stats_tools import media_por_grupo
    No exporta nada por sí mismo: cada herramienta se importa en su agente.

CUÁNDO SE EJECUTA:
    Al importar cualquier módulo de esta carpeta (lo hacen los agentes al
    construirse, es decir, al arrancar el chatbot).
"""
