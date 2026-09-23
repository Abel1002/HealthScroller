"""CLI de HealthScroller: chat en la terminal con el equipo de agentes.

Uso (desde la raíz del proyecto):
    python -m health_scroller.cli        # o el comando `healthscroller`
"""

import json
import sys
import urllib.request

from langchain_core.messages import HumanMessage

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL
from .graph import crear_grafo

# Consolas Windows (cp1252) no soportan emojis: mostramos texto sin romper el chat
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def comprobar_ollama() -> None:
    """Avisa al alumno si Ollama no está listo (sin bloquear la ejecución)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as r:
            modelos = [m.get("name", "") for m in json.load(r).get("models", [])]
        if not any(nombre.split(":")[0] == OLLAMA_MODEL.split(":")[0] for nombre in modelos):
            print(f"AVISO: el modelo '{OLLAMA_MODEL}' no está descargado. Ejecuta: ollama pull {OLLAMA_MODEL}")
    except Exception:
        print(f"AVISO: no hay respuesta de Ollama en {OLLAMA_BASE_URL} (¿ejecutaste 'ollama serve'?)")


BANNER = f"""
====================================================
  HealthScroller - Chatbot multi-agente
  Modelo: {OLLAMA_MODEL} (Ollama: {OLLAMA_BASE_URL})
  Agentes: conversacion | estadistico | algebraico | simulacion
  Comandos: /ayuda  /agente <nombre>  /reset  /salir
====================================================
"""


def _ayuda() -> str:
    return (
        "\nComandos:\n"
        "  /ayuda            muestra esta ayuda\n"
        "  /agente <nombre>  obliga a responder a un agente (conversacion, estadistico, algebraico, simulacion)\n"
        "  /reset            borra el historial de la conversación\n"
        "  /salir            cierra el chat\n"
        "Cualquier otro texto se envía al orquestador, que elige al especialista."
    )


def main() -> None:
    print(BANNER)
    comprobar_ollama()

    grafo, agentes = crear_grafo()
    historial: list = []

    while True:
        try:
            entrada = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not entrada:
            continue

        comando = entrada.lower()
        if comando in {"/salir", "/exit", "/q"}:
            print("¡Hasta pronto!")
            break
        if comando == "/ayuda":
            print(_ayuda())
            continue
        if comando == "/reset":
            historial = []
            print("Historial borrado.")
            continue
        if comando.startswith("/agente"):
            partes = entrada.split(maxsplit=1)
            nombre = partes[1].strip().lower() if len(partes) > 1 else ""
            if nombre not in agentes:
                print(f"Agentes disponibles: {', '.join(agentes)}")
                continue
            try:
                salida = agentes[nombre].invoke({"messages": [HumanMessage(content=entrada)]})
                print(f"\n[{nombre}] {salida['messages'][-1].content}")
            except Exception as error:
                print(f"Error con el agente '{nombre}': {error}")
            continue

        # Flujo normal: el orquestador decide quién responde
        historial.append(HumanMessage(content=entrada))
        try:
            estado = grafo.invoke({"messages": historial})
            historial = estado["messages"]
            print(f"\n[{estado['ruta']}] {historial[-1].content}")
        except Exception as error:
            historial.pop()
            print(f"Error al procesar la pregunta: {error}")
            print("(¿Está Ollama arrancado? Ejecuta 'ollama serve' y prueba de nuevo)")


if __name__ == "__main__":
    main()
