"""Configuración central del proyecto: rutas y parámetros de Ollama.

Los valores se pueden sobrescribir con un archivo .env (ver .env.example).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Raíz del proyecto: src/health_scroller/config.py -> subimos 3 niveles
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Recursos que usan los agentes
DATA_PATH = PROJECT_ROOT / "data" / "Social_media_impact_on_life.csv"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "analysis_results.json"

# Ollama (LLM local y gratuito)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

# Temperatura baja = respuestas más serias y repetibles
TEMPERATURA = float(os.getenv("TEMPERATURA", "0.2"))

# Ventana de contexto para los prompts (el briefing del notebook entra aquí)
NUM_CTX = 8192
