"""Grafo principal (LangGraph): un orquestador que deriva a los especialistas.

    START -> orquestador -> conversacion | estadistico | algebraico | simulacion -> END

Cada especialista es un mini-agente ReAct creado con create_react_agent:
recibe el historial, decide si llama a sus herramientas y redacta la respuesta.
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from .agents.algebra import crear_agente_algebraico
from .agents.conversation import crear_agente_conversacion
from .agents.orchestrator import clasificar_intencion
from .agents.simulation import crear_agente_simulacion
from .agents.statistics import crear_agente_estadistico


class EstadoChat(MessagesState):
    """Estado compartido del grafo: historial de mensajes + ruta elegida."""

    ruta: str


def nodo_orquestador(estado: EstadoChat) -> dict:
    """Busca la última pregunta del usuario y decide qué agente responde."""
    ultima_pregunta = ""
    for mensaje in reversed(estado["messages"]):
        if isinstance(mensaje, HumanMessage):
            ultima_pregunta = str(mensaje.content)
            break
    return {"ruta": clasificar_intencion(ultima_pregunta)}


def _enrutar(estado: EstadoChat) -> Literal["conversacion", "estadistico", "algebraico", "simulacion"]:
    return estado["ruta"]  # type: ignore[return-value]


def crear_grafo():
    """Crea el grafo compilado y el diccionario de agentes (reutilizado por la CLI)."""
    agentes = {
        "conversacion": crear_agente_conversacion(),
        "estadistico": crear_agente_estadistico(),
        "algebraico": crear_agente_algebraico(),
        "simulacion": crear_agente_simulacion(),
    }

    builder = StateGraph(EstadoChat)
    builder.add_node("orquestador", nodo_orquestador)
    for nombre, agente in agentes.items():
        builder.add_node(nombre, agente)

    builder.add_edge(START, "orquestador")
    builder.add_conditional_edges("orquestador", _enrutar, list(agentes.keys()))
    for nombre in agentes:
        builder.add_edge(nombre, END)

    return builder.compile(), agentes
