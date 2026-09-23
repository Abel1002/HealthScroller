"""Grafo principal (LangGraph): un orquestador que deriva a los especialistas.

QUÉ HACE:
    Construye el grafo de estados del chat: START -> orquestador ->
    (conversacion | estadistico | algebraico | simulacion) -> END.
    Devuelve (grafo compilado, diccionario de agentes).

PARA QUÉ SIRVE:
    Es el "cerebro de rutas" del sistema: decide qué nodo procesa cada
    pregunta. El diccionario de agentes que devuelve lo reutiliza la CLI para
    invocar agentes directamente con /agente <nombre>.

CUÁNDO SE EJECUTA:
    Una vez al arrancar el chatbot (cli.py llama a crear_grafo()); después,
    grafo.invoke() corre en CADA pregunta del usuario.
"""

from typing import Literal  # anotación de tipos: las rutas válidas como texto literal

from langchain_core.messages import HumanMessage  # mensaje del usuario (para extraer la última pregunta)
from langgraph.graph import END, START, MessagesState, StateGraph  # primitivas del grafo: nodos, aristas y estado

from .agents.algebra import crear_agente_algebraico  # fábrica del agente algebraico
from .agents.conversation import crear_agente_conversacion  # fábrica del agente conversacional
from .agents.orchestrator import clasificar_intencion  # clasificador de etiquetas (plan A + plan B)
from .agents.simulation import crear_agente_simulacion  # fábrica del agente de simulación
from .agents.statistics import crear_agente_estadistico  # fábrica del agente estadístico


class EstadoChat(MessagesState):
    """Estado compartido del grafo: historial de mensajes + ruta elegida."""

    ruta: str  # campo extra: "estadistico", "algebraico"... lo escribe el orquestador y lo lee _enrutar


def nodo_orquestador(estado: EstadoChat) -> dict:
    """Busca la última pregunta del usuario y decide qué agente responde."""
    ultima_pregunta = ""  # acumulador de la pregunta a clasificar
    for mensaje in reversed(estado["messages"]):  # recorre el historial DEL REVÉS (lo más reciente primero)
        if isinstance(mensaje, HumanMessage):  # ¿es un mensaje del usuario (no del bot)?
            ultima_pregunta = str(mensaje.content)  # nos quedamos con esa pregunta...
            break  # ...y paramos: solo interesa la última
    return {"ruta": clasificar_intencion(ultima_pregunta)}  # clasifica (LLM o palabras clave) y guarda la etiqueta en el estado


def _enrutar(estado: EstadoChat) -> Literal["conversacion", "estadistico", "algebraico", "simulacion"]:
    return estado["ruta"]  # type: ignore[return-value]  # devuelve la etiqueta que puso el orquestador (LangGraph la usará como destino)


def crear_grafo():
    """Crea el grafo compilado y el diccionario de agentes (reutilizado por la CLI)."""
    # Diccionario CENTRAL del equipo: cada clave es un nodo/ruta/etiqueta del
    # sistema; añadir una entrada aquí añade el agente a TODO (grafo + CLI)
    agentes = {
        "conversacion": crear_agente_conversacion(),  # agente sin tools (saludos/guía)
        "estadistico": crear_agente_estadistico(),  # 3 tools de pandas (descriptiva)
        "algebraico": crear_agente_algebraico(),  # 3 tools de numpy (correlación/recta)
        "simulacion": crear_agente_simulacion(),  # 3 tools de muestreo (MC/bootstrap)
    }

    builder = StateGraph(EstadoChat)  # constructor del grafo: define el tipo de estado que circula
    builder.add_node("orquestador", nodo_orquestador)  # nodo 1: clasificador (función normal, sin LLM propio... bueno, su LLM está dentro)
    for nombre, agente in agentes.items():  # registra los 4 especialistas como nodos...
        builder.add_node(nombre, agente)  # ...cada AgenteHerramientas es invocable como nodo (tiene __call__)

    builder.add_edge(START, "orquestador")  # arista fija: toda pregunta ENTRA primero por el orquestador
    builder.add_conditional_edges("orquestador", _enrutar, list(agentes.keys()))  # bifurcación: destino = estado["ruta"] (las opciones salen del dict)
    for nombre in agentes:
        builder.add_edge(nombre, END)  # cada especialista, al terminar, va a END (fin de la ejecución)

    return builder.compile(), agentes  # compila (grafo ejecutable) y devuelve también el dict para la CLI
