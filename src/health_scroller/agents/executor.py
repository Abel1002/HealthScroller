"""Ejecutor de agentes con herramientas compatibles con modelos sin tool-calling nativo.

QUÉ HACE:
    Implementa la clase AgenteHerramientas: el "cerebro" que convierte a cada
    especialista en un agente ReAct completo SIN depender del tool-calling
    nativo de Ollama. Flujo: el modelo pide un JSON con una herramienta ->
    Python la ejecuta -> se le devuelve el resultado -> el modelo redacta la
    respuesta final en texto normal.

PARA QUÉ SIRVE:
    gemma3:1b (Ollama) no soporta function calling nativo, así que sin este
    bucle los agentes no podrían consultar pandas/numpy. Además garantiza que
    los números SIEMPRE salen de los datos, nunca de la imaginación del LLM.

CUÁNDO SE EJECUTA:
    1) Al construir cada agente (se arma su prompt con la sección de tools).
    2) En CADA pregunta que recibe un especialista: su __call__/invoke lanza
    el bucle de hasta MAX_PASOS iteraciones hasta conseguir la respuesta.
"""

import json  # librería estándar: parsear el JSON que escribe el modelo
import re  # librería estándar: limpiar los bloques ```json que a veces añade el modelo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # roles de mensaje: sistema/usuario/asistente
from langchain_ollama import ChatOllama  # cliente LLM de Ollama (langchain-ollama)

from ..config import NUM_CTX, OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURA  # configuración global compartida

# Máximo de pasos herramienta->resultado antes de forzar una respuesta
# Red de seguridad: evita bucles infinitos si el modelo no acierta a responder
MAX_PASOS = 4

# Claves que el modelo puede usar en su JSON: aceptamos varias por tolerancia
# (gemma3 a veces usa "tool"/"args" en vez de "herramienta"/"argumentos")
CLAVES_NOMBRE = ("herramienta", "tool", "nombre", "name")
CLAVES_ARGUMENTOS = ("argumentos", "args", "arguments")


def construir_seccion_herramientas(tools: list) -> str:
    """Describe en el prompt qué herramientas existen y cómo invocarlas."""
    if not tools:  # agente sin tools (p. ej. conversacion): no se añade sección
        return ""
    lineas = [  # líneas base de la sección que se pegará al final del prompt
        "## Herramientas disponibles",  # título de la sección
        "Cuando necesites calcular algo, responde EXCLUSIVAMENTE con un JSON de este tipo (sin texto alrededor):",  # instrucción de formato
        '{"herramienta": "<nombre>", "argumentos": {"<param>": "<valor>"}}',  # plantilla exacta del JSON pedido
        "- Una herramienta por mensaje.",  # limita a 1 tool por paso: simplifica el bucle
        "- Cuando ya no necesites calcular, responde con texto normal (sin JSON).",  # condición de SALIDA del bucle
        "",  # línea en blanco de separación antes del listado
    ]
    for tool in tools:  # por cada herramienta, describimos su firma
        esquema = tool.tool_call_schema.model_json_schema()  # esquema JSON generado por LangChain a partir de la función
        props = esquema.get("properties", {})  # diccionario de parámetros {nombre: {type, default...}}
        requeridos = set(esquema.get("required", []))  # conjunto de parámetros obligatorios
        params = []  # acumulador de "nombre: tipo" para pintar la firma
        for nombre, info in props.items():  # recorre cada parámetro del esquema
            tipo = info.get("type", "?")  # tipo esperado (string, number, integer...)
            defecto = info.get("default")  # valor por defecto si lo tiene
            marca = f" (por defecto {defecto})" if defecto is not None else ("" if nombre in requeridos else " (opcional)")  # anota defaults u opcionales
            params.append(f"{nombre}: {tipo}{marca}")  # p. ej. "columna: string (opcional)"
        lineas.append(f"- {tool.name}({', '.join(params)}) -> {tool.description}")  # línea final: firma + docstring (lo que "ve" el LLM)
    return "\n".join(lineas)  # une todo con saltos de línea


def extraer_llamada_json(texto: str) -> tuple[str, dict] | None:
    """Busca un JSON con nombre de herramienta en la respuesta del modelo."""
    limpio = re.sub(r"```(?:json)?", "", texto, flags=re.IGNORECASE).strip("` \n")  # elimina bloques de código ```/```json que rodean el JSON
    inicio = limpio.find("{")  # posición de la primera llave de apertura
    if inicio == -1:  # ¿ni siquiera hay "{"?
        return None  # None = el modelo respondió texto normal (sin pedir tool)

    # Localizamos la llave de cierre que empareja la primera de apertura
    # Parser manual de llaves balanceadas (json.loads solo no basta si hay
    # texto antes/después del JSON)
    profundidad = 0  # contador de llaves abiertas { ... }
    en_cadena = False  # ¿estamos dentro de un "string"? (los { de dentro no cuentan)
    escapar = False  # ¿el carácter anterior fue \ ? (para \" no confundir comillas)
    fin = -1  # posición de la llave de cierre que buscamos (-1 = no encontrada)
    for posicion, caracter in enumerate(limpio[inicio:], start=inicio):  # recorre desde la primera {
        if escapar:  # carácter escapado anteriormente (\x): se ignora
            escapar = False
            continue
        if caracter == "\\":  # barra invertida: el siguiente carácter es literal
            escapar = True
            continue
        if caracter == '"':  # comilla: alternamos dentro/fuera de cadena
            en_cadena = not en_cadena
            continue
        if en_cadena:  # dentro de un string no interpretamos { } "
            continue
        if caracter == "{":  # llave fuera de cadena...
            profundidad += 1  # ...sube el nivel
        elif caracter == "}":  # llave de cierre fuera de cadena...
            profundidad -= 1  # ...baja el nivel
            if profundidad == 0:  # ¿volvimos al nivel 0? -> JSON completo
                fin = posicion + 1  # fin = índice AFTER de la última }
                break
    if fin == -1:  # llaves sin cerrar (JSON truncado)
        return None  # no se pudo extraer: tratamos como texto normal

    try:
        datos = json.loads(limpio[inicio:fin])  # parsea el trozo { ... } extraído
    except json.JSONDecodeError:
        return None  # JSON mal formado -> mejor responder como texto
    if not isinstance(datos, dict):
        return None  # el JSON no es un objeto (p. ej. una lista) -> inútil para pedir tools

    nombre = next((datos[clave] for clave in CLAVES_NOMBRE if clave in datos), None)  # busca el nombre bajo cualquier clave aceptada
    if not isinstance(nombre, str) or not nombre.strip():
        return None  # no había nombre de herramienta válido -> no es una llamada

    argumentos = next((datos[clave] for clave in CLAVES_ARGUMENTOS if clave in datos), {})  # busca los argumentos (por defecto {})
    if not isinstance(argumentos, dict):
        argumentos = {}  # si venían en formato raro, usamos {} para no romper tool.invoke
    return nombre.strip(), argumentos  # (nombre, argumentos) listos para ejecutar


def _preparar_argumentos(tool, argumentos: dict) -> dict:
    """Convierte números enviados como texto ("3.5") al tipo que espera la herramienta."""
    esquema = tool.tool_call_schema.model_json_schema()  # esquema de la tool (tipos esperados)
    props = esquema.get("properties", {})  # {parametro: tipo...}
    preparados = {}  # copia corregida de los argumentos
    for clave, valor in argumentos.items():  # revisa cada argumento que mandó el modelo
        tipo = props.get(clave, {}).get("type")  # tipo que espera la función
        if isinstance(valor, str) and tipo in ("number", "integer"):  # el modelo mandó "3.5" (texto) y toca número
            try:
                numero = float(valor)  # "3.5" -> 3.5
                preparados[clave] = int(numero) if tipo == "integer" else numero  # 3 -> int, number -> float
                continue  # argumento arreglado: pasamos al siguiente
            except ValueError:
                pass  # no era convertible: lo dejamos tal cual (la tool dirá su error)
        preparados[clave] = valor  # argumento ya correcto: se copia sin tocar
    return preparados  # argumentos listos para tool.invoke()


class AgenteHerramientas:
    """Agente con bucle de herramientas compatible con cualquier modelo de Ollama.

    Expone la misma interfaz que los agentes de LangGraph: invoke(state) -> dict.
    """

    def __init__(self, prompt: str, tools: list):  # prompt = instrucciones del especialista (sin la sección de tools aún)
        self.tools = {tool.name: tool for tool in tools}  # diccionario {nombre: tool} para buscarlas rápido en el bucle
        self.llm = ChatOllama(  # cliente del modelo local para ESTE agente
            model=OLLAMA_MODEL,  # p. ej. gemma3:1b (de config.py)
            base_url=OLLAMA_BASE_URL,  # servidor Ollama (localhost:11434 por defecto)
            temperature=TEMPERATURA,  # 0.2: respuestas serias y casi repetibles
            num_ctx=NUM_CTX,  # 8192 tokens: cabe el briefing + historial
        )
        seccion = construir_seccion_herramientas(list(self.tools.values()))  # genera el listado de firmas de tools
        self.prompt = f"{prompt}\n\n{seccion}" if seccion else prompt  # prompt final = especialista + sección de herramientas

    def __call__(self, state: dict) -> dict:  # LangGraph invoca los nodos como si fueran funciones
        return self.invoke(state)  # redirigimos a invoke: misma lógica, dos formas de llamar

    def invoke(self, state: dict) -> dict:
        """Devuelve los mensajes nuevos generados en este turno (solo la respuesta final)."""
        return {"messages": self._responder(state["messages"])}  # estado con SOLO el mensaje nuevo (LangGraph lo añade al historial)

    def _responder(self, mensajes_previos: list) -> list:
        mensajes_llm = [SystemMessage(content=self.prompt), *mensajes_previos]  # prompt de sistema (reglas+tools) + toda la conversación

        for _ in range(MAX_PASOS):  # bucle ReAct: máximo 4 idas y vueltas LLM <-> herramientas
            respuesta = self.llm.invoke(mensajes_llm)  # 1) preguntamos al LLM
            llamada = extraer_llamada_json(str(respuesta.content))  # 2) ¿respondió un JSON con herramienta?

            if llamada is None:
                return [respuesta]  # no pidió tool -> texto final: este mensaje cierra el turno

            nombre, argumentos = llamada  # desempaqueta la petición: {"herramienta": ..., "argumentos": ...}
            tool = self.tools.get(nombre)  # ¿existe esa herramienta en este agente?
            if tool is None:
                disponibles = ", ".join(self.tools)  # lista de nombres válidos
                resultado = f"Error: la herramienta '{nombre}' no existe. Disponibles: {disponibles}"  # error instructivo para que el LLM corrija
            else:
                try:
                    resultado = tool.invoke(_preparar_argumentos(tool, argumentos))  # ejecuta la función Python REAL (pandas/numpy) con tipos corregidos
                except Exception as error:
                    # El error se devuelve como instrucción: el modelo debe reintentar
                    # con los nombres EXACTOS de parámetros del listado
                    firmas = construir_seccion_herramientas([tool])  # regenera la firma correcta de ESA tool
                    resultado = (  # mensaje de auto-corrección que volverá al LLM
                        f"Error al ejecutar '{nombre}': {error}\n\n"
                        f"{firmas}\n\n"
                        "Los nombres de los argumentos deben coincidir EXACTAMENTE con "
                        "los del listado. Responde de nuevo SOLO con el JSON corregido."
                    )

            # El resultado se devuelve como mensaje de usuario: así funciona
            # con cualquier plantilla de Ollama (sin necesidad de tool-calling nativo)
            mensajes_llm.append(respuesta)  # guardamos lo que dijo el LLM (petición en JSON)
            mensajes_llm.append(
                HumanMessage(
                    content=(
                        f"RESULTADO de la herramienta '{nombre}':\n{resultado}\n\n"  # qué tool y qué devolvió
                        "Usa este resultado para responder al usuario en español "  # instrucción de redacción final
                        "con texto normal (sin JSON)."
                    )
                )
            )  # y el bucle vuelve al inicio: ahora el LLM ya tiene el dato para responder

        return [
            AIMessage(  # se agotaron los 4 pasos sin texto final: pedimos reformular
                content="Necesité varias consultas y aún no tengo un resultado claro. "
                "¿Puedes simplificar o reformular la pregunta?"
            )
        ]
