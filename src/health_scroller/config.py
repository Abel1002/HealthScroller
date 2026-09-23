"""Configuración central del proyecto: rutas y parámetros de Ollama.

QUÉ HACE:
    Define TODOS los valores globales del sistema: dónde están los datos,
    qué modelo de Ollama se usa, a qué servidor se conecta y con qué temperatura.

PARA QUÉ SIRVE:
    Es el módulo que importa casi todo el resto: si quieres cambiar de modelo
    o de temperatura, solo tocas este fichero (o un archivo .env), sin buscar
    por todo el código.

CUÁNDO SE EJECUTA:
    UNA sola vez, en el primer import (arranque del chatbot o de cualquier
    tool); sus constantes quedan en memoria durante toda la ejecución.
"""

import os  # librería estándar: lee variables de entorno del sistema operativo
from pathlib import Path  # librería estándar: rutas de ficheros multiplataforma (Windows/Mac/Linux)

from dotenv import load_dotenv  # python-dotenv: carga variables desde un fichero .env

load_dotenv()  # Si existe un .env, sus variables pasan a os.getenv: mandan sobre los valores por defecto de abajo

# Raíz del proyecto: src/health_scroller/config.py -> subimos 3 niveles
# parents[0]=health_scroller, parents[1]=src, parents[2]=raíz (donde viven data/ y outputs/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ruta absoluta al dataset CSV: la usa knowledge.cargar_dataset() y, con ella, TODAS las tools
DATA_PATH = PROJECT_ROOT / "data" / "Social_media_impact_on_life.csv"
# Ruta al briefing JSON que genera el notebook: lo leen los prompts de los agentes
RESULTS_PATH = PROJECT_ROOT / "outputs" / "analysis_results.json"

# --- Ollama (LLM local y gratuito) ---
# URL del servidor Ollama: por defecto, esta misma máquina en el puerto 11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Modelo que responderá: gemma3:1b (pequeño, rápido, gratis); override posible por .env
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

# Temperatura baja = respuestas más serias y repetibles
# Temperatura del LLM de los especialistas: 0.2 ≈ casi determinista (en .env se puede cambiar)
TEMPERATURA = float(os.getenv("TEMPERATURA", "0.2"))  # float(): getenv devuelve texto y hay que convertirlo

# Ventana de contexto para los prompts (el briefing del notebook entra aquí)
# Nº máximo de "tokens" que el modelo admite por llamada; 8192 cabe el briefing completo
NUM_CTX = 8192
