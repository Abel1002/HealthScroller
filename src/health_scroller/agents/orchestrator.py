"""Agente orquestador: lee la pregunta del usuario y elige qué especialista responde.

QUÉ HACE:
    Clasifica CADA pregunta en una de 4 etiquetas (conversacion, estadistico,
    algebraico, simulacion) con una estrategia doble:
    1. Plan A: le pide al LLM que responda SOLO la etiqueta.
    2. Plan B: si el LLM falla, busca palabras clave con Python puro.

PARA QUÉ SIRVE:
    Es el "recepcionista" del equipo: sin él, el usuario tendría que elegir a
    mano el agente con /agente. Su etiqueta es la `ruta` que LangGraph usa para
    enrutar la pregunta al nodo correcto del grafo.

CUÁNDO SE EJECUTA:
    En CADA turno de chat normal (no con /agente): graph.py llama a
    clasificar_intencion() desde el nodo_orquestador antes de derivar.
"""

import unicodedata  # librería estándar: normalizar texto (quitar tildes, etc.)

from langchain_core.messages import HumanMessage, SystemMessage  # tipos de mensaje de LangChain
from langchain_ollama import ChatOllama  # cliente LLM de Ollama (langchain-ollama)

from ..config import NUM_CTX, OLLAMA_BASE_URL, OLLAMA_MODEL  # parámetros globales de config.py

# Orden de prioridad: de lo más específico a lo más general
# Este orden IMPORTA: en el plan B se recorre en este sentido, así que si una
# pregunta casa con dos etiquetas, gana la aparezca antes (p. ej. "simulacion"
# antes que "estadistico")
ETIQUETAS = ("simulacion", "algebraico", "estadistico", "conversacion")

# Prompt del plan A. Van TODO dentro del string (los comentarios NO se ponen
# dentro: se enviarían al LLM como parte del prompt). Estructura:
# 1) identidad del rol, 2) lista de etiquetas con descripción, 3) ejemplos
# few-shot (enseñanza por ejemplo), 4) orden de responder SOLO la etiqueta.
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

Responde SOLO con la etiqueta, sin explicaciones."""  # el fragmento final es clave: evita que el modelo razonamiento largo

# Plan B: si el LLM se equivoca, decidimos con palabras clave simples
# Diccionario etiqueta -> palabras que, si aparecen en la pregunta, la clasifican
# en esa etiqueta SIN necesidad de LLM (rápido, gratis y determinista)
PALABRAS_CLAVE = {
    "estadistico": ["media", "mediana", "promedio", "cuantos", "cuántos", "conteo", "resumen", "grupo", "por género", "por plataforma", "por nivel"],
    "algebraico": ["correlac", "regresi", "pendiente", "predic", "recta", "r2"],
    "simulacion": ["monte carlo", "bootstrap", "intervalo", "simula", "pasaría si", "pasaria si", "escenario"],
}


def _normalizar(texto: str) -> str:
    """Quita mayúsculas y acentos para comparar sin errores."""
    sin_tildes = unicodedata.normalize("NFD", texto.lower())  # lower() pasa a minúsculas; NFD separa cada letra de sus tildes
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")  # Mn = marcas combinantes (tilde): se descartan, "cuál" -> "cual"


def clasificar_por_palabras(mensaje: str) -> str:
    """Clasificación por palabras clave (sin LLM). Útil como red de seguridad."""
    texto = _normalizar(mensaje)  # mensaje en minúsculas y sin tildes
    for etiqueta in ETIQUETAS:  # recorre las etiquetas EN ORDEN de prioridad
        palabras = PALABRAS_CLAVE.get(etiqueta, [])  # palabras de esa etiqueta ([] si no tiene: p. ej. conversacion)
        if any(_normalizar(palabra) in texto for palabra in palabras):  # ¿aparece alguna palabra de la lista?
            return etiqueta  # primera coincidencia gana (respeta la prioridad)
    return "conversacion"  # red de seguridad final: sin coincidencias, charla general


def clasificar_intencion(mensaje: str) -> str:
    """Pregunta al LLM por una etiqueta y la valida; si falla, usa el plan B."""
    llm = ChatOllama(  # cliente del modelo local...
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0, num_ctx=NUM_CTX  # ...con temperatura 0.0: máxima estabilidad para clasificar
    )
    try:  # plan A dentro de un try: cualquier fallo (Ollama caído, respuesta rara) no rompe el chat
        respuesta = llm.invoke(  # llamada al LLM: prompt del sistema (reglas) + la pregunta del usuario
            [SystemMessage(content=PROMPT_ORQUESTADOR), HumanMessage(content=mensaje)]
        )
        texto = _normalizar(str(respuesta.content))  # respuesta en minúsculas/sin tildes (str: el contenido puede ser un objeto mensaje)
        for etiqueta in ETIQUETAS:  # validamos: ¿la respuesta CONTIENE alguna etiqueta válida?
            if etiqueta in texto:
                return etiqueta  # plan A con éxito: devolvemos la primera etiqueta encontrada
    except Exception:
        pass  # si Ollama no responde, caemos al plan B (sin imprimir errores al alumno)
    return clasificar_por_palabras(mensaje)  # plan B: palabras clave con Python puro
