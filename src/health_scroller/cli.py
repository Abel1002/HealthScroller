"""CLI de HealthScroller: chat en la terminal con el equipo de agentes.

QUÉ HACE:
    Interfaz de línea de comandos del chat: muestra el banner, comprueba que
    Ollama esté vivo, carga el grafo y ejecuta el bucle de conversación
    (leer pregunta -> grafo -> imprimir respuesta) con comandos /ayuda,
    /agente, /reset y /salir.

PARA QUÉ SIRVE:
    Es el PUNTO DE ENTRADA del programa: lo que el alumno ejecuta para hablar
    con los agentes (python -m health_scroller.cli o el comando `healthscroller`).

CUÁNDO SE EJECUTA:
    A demanda del usuario, desde la terminal, con el entorno ya preparado.
"""

import json  # para leer la respuesta JSON del servidor Ollama (/api/tags)
import sys  # para ajustar la codificación de la salida en Windows
import urllib.request  # petición HTTP ligera a Ollama (sin dependencias extra)

from langchain_core.messages import HumanMessage  # mensajes del usuario para invocar el grafo

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL  # configuración global de config.py

# Consolas Windows (cp1252) no soportan emojis: mostramos texto sin romper el chat
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # salida en UTF-8; caracteres raros -> "?" en vez de crash
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # idem para errores
except Exception:
    pass  # si la consola no permite reconfigurar, seguimos con la codificación por defecto


def comprobar_ollama() -> None:
    """Avisa al alumno si Ollama no está listo (sin bloquear la ejecución)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as r:  # GET /api/tags = lista de modelos; timeout 3s para no colgar
            modelos = [m.get("name", "") for m in json.load(r).get("models", [])]  # nombres de los modelos instalados
        if not any(nombre.split(":")[0] == OLLAMA_MODEL.split(":")[0] for nombre in modelos):  # ¿está gemma3 (con o sin :tag)?
            print(f"AVISO: el modelo '{OLLAMA_MODEL}' no está descargado. Ejecuta: ollama pull {OLLAMA_MODEL}")  # aviso amable con la solución
    except Exception:
        print(f"AVISO: no hay respuesta de Ollama en {OLLAMA_BASE_URL} (¿ejecutaste 'ollama serve'?)")  # servidor caído: decimos cómo levantarlo


BANNER = f"""
===================================================
  HealthScroller - Chatbot multi-agente
  Modelo: {OLLAMA_MODEL} (Ollama: {OLLAMA_BASE_URL})
  Agentes: conversacion | estadistico | algebraico | simulacion
  Comandos: /ayuda  /agente <nombre>  /reset  /salir
===================================================
"""  # cabecera impresa al arrancar (f-string: {..} se sustituyen por config)


def _ayuda() -> str:
    return (  # texto del comando /ayuda (siempre en ASCII seguro para consolas Windows)
        "\nComandos:\n"
        "  /ayuda            muestra esta ayuda\n"
        "  /agente <nombre>  obliga a responder a un agente (conversacion, estadistico, algebraico, simulacion)\n"
        "  /reset            borra el historial de la conversación\n"
        "  /salir            cierra el chat\n"
        "Cualquier otro texto se envía al orquestador, que elige al especialista."
    )


def main() -> None:
    print(BANNER)  # 1) identidad + comandos del chat
    comprobar_ollama()  # 2) avisa si Ollama/modelo no están listos (no corta: se puede seguir para ver errores)

    # Los prompts de los agentes necesitan el briefing del notebook.
    # Si falta, se avisa con claridad en vez de mostrar un traceback.
    try:
        from .graph import crear_grafo  # import local: solo aquí, para poder capturar su FileNotFoundError

        grafo, agentes = crear_grafo()  # construye prompts (necesita el JSON), tools y grafo compilado
    except FileNotFoundError as error:  # falta outputs/analysis_results.json (no ejecutó el notebook)
        print(f"\nERROR: {error}")  # mensaje claro de knowledge.cargar_briefing()
        print("\nFalta outputs/analysis_results.json (lo genera el notebook).")
        print("Ejecuta primero el paso 9 del README:")
        print("  notebooks/clase2_health_scroller_analisis.ipynb -> Kernel > Restart & Run All")
        print("Y vuelve a lanzar el chatbot.")
        return  # salimos SIN traceback y SIN bucle de chat

    historial: list = []  # mensajes Human/AI acumulados entre turnos (estado de la conversación)

    while True:  # bucle principal del chat: se repite hasta /salir o Ctrl+C
        try:
            entrada = input("\nTú > ").strip()  # lee lo que escribe el alumno y quita espacios
        except (EOFError, KeyboardInterrupt):  # Ctrl+C o cierre de la terminal
            print()  # salto de línea estético
            break  # termina el programa limpiamente

        if not entrada:  # pulsó Enter sin escribir nada
            continue  # no hace nada: vuelve a preguntar

        comando = entrada.lower()  # comandos en minúsculas para comparar sin errores
        if comando in {"/salir", "/exit", "/q"}:  # ¿quiere salir?
            print("¡Hasta pronto!")  # despedida
            break  # rompe el while -> fin del programa
        if comando == "/ayuda":  # ¿pide ayuda?
            print(_ayuda())  # imprime la lista de comandos
            continue  # vuelta al inicio del bucle (no procesa como pregunta)
        if comando == "/reset":  # ¿quiere olvidar la conversación?
            historial = []  # lista vacía = el siguiente turno no recordará nada
            print("Historial borrado.")  # confirmación
            continue  # vuelta al inicio
        if comando.startswith("/agente"):  # ¿fuerza a un agente concreto? (/agente estadistico)
            partes = entrada.split(maxsplit=1)  # divide: ["/agente", "estadistico"]
            nombre = partes[1].strip().lower() if len(partes) > 1 else ""  # nombre pedido (o vacío si no había)
            if nombre not in agentes:  # ¿existe ese agente en el diccionario del grafo?
                print(f"Agentes disponibles: {', '.join(agentes)}")  # le mostramos los válidos
                continue  # no hacemos nada más
            try:
                salida = agentes[nombre].invoke({"messages": [HumanMessage(content=entrada)]})  # invoca SOLO ese agente (sin pasar por orquestador)
                print(f"\n[{nombre}] {salida['messages'][-1].content}")  # imprime su respuesta marcando quién habla
            except Exception as error:
                print(f"Error con el agente '{nombre}': {error}")  # error de ese agente no rompe el chat
            continue

        # Flujo normal: el orquestador decide quién responde
        historial.append(HumanMessage(content=entrada))  # añadimos la pregunta al historial (candidato provisional)
        try:
            estado = grafo.invoke({"messages": historial})  # recorre TODO el grafo: orquestador -> especialista -> END
            historial = estado["messages"]  # guardamos el historial ampliado (ahora incluye la respuesta del bot)
            print(f"\n[{estado['ruta']}] {historial[-1].content}")  # mostramos [etiqueta] + respuesta final
        except Exception as error:
            historial.pop()  # DESHACEMOS la pregunta: si falló, no debe quedarse en el historial
            print(f"Error al procesar la pregunta: {error}")  # error visible para depurar
            print("(¿Está Ollama arrancado? Ejecuta 'ollama serve' y prueba de nuevo)")  # pista más frecuente de fallo


if __name__ == "__main__":  # si ejecutan este fichero directamente (python cli.py / -m)
    main()  # arranca la CLI
