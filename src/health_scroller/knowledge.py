"""Carga de recursos: el dataset (CSV) y el briefing generado por el notebook.

Los agentes se nutren de dos fuentes:
1. El dataset original (data/*.csv), consultado en vivo con las herramientas.
2. El resumen precalculado (outputs/analysis_results.json) que exporta el
   notebook de la clase 2.
"""

import functools
import json

import pandas as pd

from .config import DATA_PATH, RESULTS_PATH

# Igual que en el notebook (sección 3.1): columnas con nulos que se imputan con la mediana
COLUMNAS_CON_NULOS = ["Perceived_Stress_Score", "Academic_Performance_GPA"]


@functools.lru_cache
def cargar_dataset() -> pd.DataFrame:
    """Devuelve el DataFrame del proyecto (se carga una sola vez y se cachea).

    Aplica la misma limpieza que el notebook: imputación de nulos con la mediana.
    Así las herramientas siempre calculan sobre datos limpios.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    for columna in COLUMNAS_CON_NULOS:
        df[columna] = df[columna].fillna(df[columna].median())
    return df


@functools.lru_cache
def cargar_briefing() -> dict:
    """Devuelve el resumen del notebook (analysis_results.json) como diccionario."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra {RESULTS_PATH}. Ejecuta primero el notebook "
            "notebooks/clase2_health_scroller_analisis.ipynb"
        )
    with RESULTS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache
def briefing_texto() -> str:
    """Briefing en formato texto (JSON legible) para inyectar en los prompts."""
    return json.dumps(cargar_briefing(), ensure_ascii=False, indent=2)
