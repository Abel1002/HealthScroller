"""Ejecutor de agentes con herramientas compatibles con modelos sin tool-calling nativo.

gemma3:1b (Ollama) no soporta function calling nativo, así que usamos un bucle
ReAct sencillo en texto:

1. El modelo responde con un JSON: {"herramienta": ..., "argumentos": ...}
2. Nosotros ejecutamos la herramienta (pandas/numpy) con Python real.
3. Le devolvemos el resultado y el modelo redacta la respuesta final.

Así los números SIEMPRE salen de los datos, nunca de la imaginación del LLM.
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ..config import NUM_CTX, OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURA

# Máximo de pasos herramienta->resultado antes de forzar una respuesta
MAX_PASOS = 4

CLAVES_NOMBRE = ("herramienta", "tool", "nombre", "name")
CLAVES_ARGUMENTOS = ("argumentos", "args", "arguments")


def construir_seccion_herramientas(tools: list) -> str:
    """Describe en el prompt qué herramientas existen y cómo invocarlas."""
    if not tools:
        return ""
    lineas = [
        "## Herramientas disponibles",
        "Cuando necesites calcular algo, responde EXCLUSIVAMENTE con un JSON de este tipo (sin texto alrededor):",
        '{"herramienta": "<nombre>", "argumentos": {"<param>": "<valor>"}}',
        "- Una herramienta por mensaje.",
        "- Cuando ya no necesites calcular, responde con texto normal (sin JSON).",
        "",
    ]
    for tool in tools:
        esquema = tool.tool_call_schema.model_json_schema()
        props = esquema.get("properties", {})
        requeridos = set(esquema.get("required", []))
        params = []
        for nombre, info in props.items():
            tipo = info.get("type", "?")
            defecto = info.get("default")
            marca = f" (por defecto {defecto})" if defecto is not None else ("" if nombre in requeridos else " (opcional)")
            params.append(f"{nombre}: {tipo}{marca}")
        lineas.append(f"- {tool.name}({', '.join(params)}) -> {tool.description}")
    return "\n".join(lineas)


def extraer_llamada_json(texto: str) -> tuple[str, dict] | None:
    """Busca un JSON con nombre de herramienta en la respuesta del modelo."""
    limpio = re.sub(r"```(?:json)?", "", texto, flags=re.IGNORECASE).strip("` \n")
    inicio = limpio.find("{")
    if inicio == -1:
        return None

    # Localizamos la llave de cierre que empareja la primera de apertura
    profundidad = 0
    en_cadena = False
    escapar = False
    fin = -1
    for posicion, caracter in enumerate(limpio[inicio:], start=inicio):
        if escapar:
            escapar = False
            continue
        if caracter == "\\":
            escapar = True
            continue
        if caracter == '"':
            en_cadena = not en_cadena
            continue
        if en_cadena:
            continue
        if caracter == "{":
            profundidad += 1
        elif caracter == "}":
            profundidad -= 1
            if profundidad == 0:
                fin = posicion + 1
                break
    if fin == -1:
        return None

    try:
        datos = json.loads(limpio[inicio:fin])
    except json.JSONDecodeError:
        return None
    if not isinstance(datos, dict):
        return None

    nombre = next((datos[clave] for clave in CLAVES_NOMBRE if clave in datos), None)
    if not isinstance(nombre, str) or not nombre.strip():
        return None

    argumentos = next((datos[clave] for clave in CLAVES_ARGUMENTOS if clave in datos), {})
    if not isinstance(argumentos, dict):
        argumentos = {}
    return nombre.strip(), argumentos


def _preparar_argumentos(tool, argumentos: dict) -> dict:
    """Convierte números enviados como texto ("3.5") al tipo que espera la herramienta."""
    esquema = tool.tool_call_schema.model_json_schema()
    props = esquema.get("properties", {})
    preparados = {}
    for clave, valor in argumentos.items():
        tipo = props.get(clave, {}).get("type")
        if isinstance(valor, str) and tipo in ("number", "integer"):
            try:
                numero = float(valor)
                preparados[clave] = int(numero) if tipo == "integer" else numero
                continue
            except ValueError:
                pass
        preparados[clave] = valor
    return preparados


class AgenteHerramientas:
    """Agente con bucle de herramientas compatible con cualquier modelo de Ollama.

    Expone la misma interfaz que los agentes de LangGraph: invoke(state) -> dict.
    """

    def __init__(self, prompt: str, tools: list):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURA,
            num_ctx=NUM_CTX,
        )
        seccion = construir_seccion_herramientas(list(self.tools.values()))
        self.prompt = f"{prompt}\n\n{seccion}" if seccion else prompt

    def __call__(self, state: dict) -> dict:
        return self.invoke(state)

    def invoke(self, state: dict) -> dict:
        """Devuelve los mensajes nuevos generados en este turno (solo la respuesta final)."""
        return {"messages": self._responder(state["messages"])}

    def _responder(self, mensajes_previos: list) -> list:
        mensajes_llm = [SystemMessage(content=self.prompt), *mensajes_previos]

        for _ in range(MAX_PASOS):
            respuesta = self.llm.invoke(mensajes_llm)
            llamada = extraer_llamada_json(str(respuesta.content))

            if llamada is None:
                return [respuesta]

            nombre, argumentos = llamada
            tool = self.tools.get(nombre)
            if tool is None:
                disponibles = ", ".join(self.tools)
                resultado = f"Error: la herramienta '{nombre}' no existe. Disponibles: {disponibles}"
            else:
                try:
                    resultado = tool.invoke(_preparar_argumentos(tool, argumentos))
                except Exception as error:
                    # El error se devuelve como instrucción: el modelo debe reintentar
                    # con los nombres EXACTOS de parámetros del listado
                    firmas = construir_seccion_herramientas([tool])
                    resultado = (
                        f"Error al ejecutar '{nombre}': {error}\n\n"
                        f"{firmas}\n\n"
                        "Los nombres de los argumentos deben coincidir EXACTAMENTE con "
                        "los del listado. Responde de nuevo SOLO con el JSON corregido."
                    )

            # El resultado se devuelve como mensaje de usuario: así funciona
            # con cualquier plantilla de Ollama (sin necesidad de tool-calling nativo)
            mensajes_llm.append(respuesta)
            mensajes_llm.append(
                HumanMessage(
                    content=(
                        f"RESULTADO de la herramienta '{nombre}':\n{resultado}\n\n"
                        "Usa este resultado para responder al usuario en español "
                        "con texto normal (sin JSON)."
                    )
                )
            )

        return [
            AIMessage(
                content="Necesité varias consultas y aún no tengo un resultado claro. "
                "¿Puedes simplificar o reformular la pregunta?"
            )
        ]
