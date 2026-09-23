"""Carga de recursos: el dataset (CSV) y el briefing generado por el notebook.

QUÉ HACE:
    Centraliza la lectura de las DOS fuentes de conocimiento del proyecto:
    1. data/*.csv                -> datos "crudos", consultados en vivo por las tools
    2. outputs/analysis_results.json -> resumen precalculado por el notebook

PARA QUÉ SIRVE:
    Todos los módulos piden los datos AQUÍ (y solo aquí): si cambia la ruta o
    la limpieza, se cambia en este fichero y el cambio afecta a todo el sistema.

CUÁNDO SE EJECUTA:
    - cargar_dataset(): la primera vez que una tool necesita calcular algo.
    - cargar_briefing(): al construir los prompts de los agentes (arranque).
    Ambas quedan en caché (lru_cache): la 2ª llamada devuelve el resultado
    guardado sin volver a leer el disco.
"""

import functools  # decorador lru_cache: memoriza el resultado de una función
import json  # librería estándar: convertir JSON <-> diccionarios de Python

import pandas as pd  # pandas: aquí para leer el CSV en un DataFrame

from .config import DATA_PATH, RESULTS_PATH  # rutas absolutas definidas en config.py (el . = paquete actual)

# Igual que en el notebook (sección 3.1): columnas con nulos que se imputan con la mediana
# Sin esta imputación, medias y polyfit darían NaN y los agentes responderían basura
COLUMNAS_CON_NULOS = ["Perceived_Stress_Score", "Academic_Performance_GPA"]


@functools.lru_cache  # caché: el CSV se lee UNA vez por ejecución; repetir la llamada es instantáneo
def cargar_dataset() -> pd.DataFrame:
    """Devuelve el DataFrame del proyecto (se carga una sola vez y se cachea).

    Aplica la misma limpieza que el notebook (sección 3.1): imputación de nulos con la mediana.
    Así las herramientas siempre calculan sobre datos limpios.
    """
    if not DATA_PATH.exists():  # ¿existe el CSV? Si no, error claro para el alumno (se vería en la CLI)
        raise FileNotFoundError(f"No se encuentra el dataset en: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)  # lee las 4500 filas del CSV a un DataFrame de pandas
    for columna in COLUMNAS_CON_NULOS:  # recorre las 2 columnas que pueden traer huecos (NaN)
        df[columna] = df[columna].fillna(df[columna].median())  # cada nulo -> MEDIANA de su columna (igual que el notebook)
    return df  # devuelve el DataFrame limpio (lru_cache lo guarda para las próximas llamadas)


@functools.lru_cache  # el JSON del notebook también se lee una sola vez
def cargar_briefing() -> dict:
    """Devuelve el resumen del notebook (analysis_results.json) como diccionario."""
    if not RESULTS_PATH.exists():  # si el alumno no ha ejecutado el notebook, avisamos con instrucciones
        raise FileNotFoundError(
            f"No se encuentra {RESULTS_PATH}. Ejecuta primero el notebook "
            "notebooks/clase2_health_scroller_analisis.ipynb"
        )
    with RESULTS_PATH.open(encoding="utf-8") as f:  # abre el JSON como texto UTF-8 (el with cierra el fichero solo)
        return json.load(f)  # parsea el texto JSON a un dict de Python


@functools.lru_cache  # la conversión dict->texto también se memoriza
def briefing_texto() -> str:
    """Briefing en formato texto (JSON legible) para inyectar en los prompts."""
    return json.dumps(cargar_briefing(), ensure_ascii=False, indent=2)  # dict -> str con tildes y sangría legible
