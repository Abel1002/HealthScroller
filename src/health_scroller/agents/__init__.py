"""Agentes del equipo: orquestador + conversación + 3 especialistas.

QUÉ HACE:
    Convierte la carpeta `agents/` en el subpaquete `health_scroller.agents`.

PARA QUÉ SIRVE:
    Permite importar los agentes con rutas limpias:
        from health_scroller.agents.statistics import crear_agente_estadistico
    Cada módulo de esta carpeta define el PROMPT de un agente y su fábrica
    `crear_agente_*()`.

CUÁNDO SE EJECUTA:
    Al importar cualquier módulo de esta carpeta (lo hace graph.py al
    construir el grafo, es decir, al arrancar el chatbot).
"""
