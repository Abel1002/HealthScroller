"""Agente orquestador: lee la pregunta del usuario y elige qué especialista responde.

Para un modelo tan pequeño como gemma3:1b usamos una trampa doble:
1. Se le pide que responda SOLO con una etiqueta.
2. Si la respuesta no es válida, hay un plan B con palabras clave.
"""

import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ..config import NUM_CTX, OLLAMA_BASE_URL, OLLAMA_MODEL

# Orden de prioridad: de lo más específico a lo más general
ETIQUETAS = ("simulacion", "algebraico", "estadistico", "conversacion")

PROMPT_ORQUESTADOR = """Eres el enrutador de un equipo de agentes. Clasifica la pregunta del usuario en UNA etiqueta:

- conversacion: saludos, despedidas, charla general o preguntas sobre el proyecto
- estadistico: medias, medianas, conteos, resúmenes o comparaciones entre grupos
- algebraico: correlaciones, regresión lineal, predicciones puntuales
- simulacion: Monte Carlo, bootstrap, intervalos de confianza, escenarios "¿qué pasaría si...?"

Ejemplos:
"¿Cuál es la media de GPA por género?" -> estadistico
"¿Qué correlación hay entre horas de uso y GPA?" -> algebraico
"¿Qué pasaría si reduzco el uso a 3 horas?" -> simulacion
"¡Hola! ¿Quién eres?" -> conversacion

Responde SOLO con la etiqueta, sin explicaciones."""

# Plan B: si el LLM se equivoca, decidimos con palabras clave simples
PALABRAS_CLAVE = {
    "estadistico": ["media", "mediana", "promedio", "cuantos", "cuántos", "conteo", "resumen", "grupo", "por género", "por plataforma", "por nivel"],
    "algebraico": ["correlac", "regresi", "pendiente", "predic", "recta", "r2"],
    "simulacion": ["monte carlo", "bootstrap", "intervalo", "simula", "pasaría si", "pasaria si", "escenario"],
}


def _normalizar(texto: str) -> str:
    """Quita mayúsculas y acentos para comparar sin errores."""
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


def clasificar_por_palabras(mensaje: str) -> str:
    """Clasificación por palabras clave (sin LLM). Útil como red de seguridad."""
    texto = _normalizar(mensaje)
    for etiqueta in ETIQUETAS:
        palabras = PALABRAS_CLAVE.get(etiqueta, [])
        if any(_normalizar(palabra) in texto for palabra in palabras):
            return etiqueta
    return "conversacion"


def clasificar_intencion(mensaje: str) -> str:
    """Pregunta al LLM por una etiqueta y la valida; si falla, usa el plan B."""
    llm = ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0, num_ctx=NUM_CTX
    )
    try:
        respuesta = llm.invoke(
            [SystemMessage(content=PROMPT_ORQUESTADOR), HumanMessage(content=mensaje)]
        )
        texto = _normalizar(str(respuesta.content))
        for etiqueta in ETIQUETAS:
            if etiqueta in texto:
                return etiqueta
    except Exception:
        pass  # si Ollama no responde, caemos al plan B
    return clasificar_por_palabras(mensaje)
