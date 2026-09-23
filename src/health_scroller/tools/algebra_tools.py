"""Herramientas algebraicas del Chatbot Algebraico (numpy + pandas).

QUÉ HACE:
    Expone 3 herramientas (@tool) para análisis de relación entre variables:
    correlación de Pearson, recta de regresión y predicción sobre esa recta.

PARA QUÉ SIRVE:
    Son la "calculadora" del agente algebraico: convierten preguntas como
    "¿qué correlación hay entre horas de uso y sueño?" en números exactos
    calculados con numpy sobre el dataset real.

CUÁNDO SE EJECUTA:
    Solo cuando el bucle del executor detecta un JSON con su nombre
    (p. ej. {"herramienta": "correlacion", "argumentos": {...}}).
"""

import numpy as np  # numpy: álgebra numérica (polyfit para ajustar la recta de regresión)
from langchain_core.tools import tool  # decorador que convierte la función en herramienta para el LLM

from ..knowledge import cargar_dataset  # DataFrame limpio con caché (.. = un nivel arriba)
from .stats_tools import COLUMNAS_NUMERICAS  # reutilizamos la misma lista blanca de columnas numéricas


def _validar_columnas(columna_a: str, columna_b: str) -> str | None:
    """Devuelve un mensaje de error si alguna columna no es numérica."""
    # El guion bajo (_) marca que es una función AUXILIAR interna, no una herramienta
    malas = [c for c in (columna_a, columna_b) if c not in COLUMNAS_NUMERICAS]  # filtra las columnas que NO están en la lista blanca
    if malas:  # si hay alguna inválida...
        return f"Error: {', '.join(malas)} no son numéricas. Usa: {', '.join(COLUMNAS_NUMERICAS)}"  # ...mensaje de error que el executor devolverá al LLM
    return None  # None = sin error: la tool puede continuar


def _interpretar(r: float) -> str:
    """Traduce el valor de r a palabras (fuerza y dirección)."""
    # Traduce el número r (-1..1) a un lenguaje que un humano entiende sin saber estadística
    fuerza = (
        "muy fuerte" if abs(r) > 0.7  # |r| casi máximo: relación casi perfecta
        else "fuerte" if abs(r) > 0.5  # relación clara
        else "moderada" if abs(r) > 0.3  # relación visible pero débil
        else "débil"  # casi sin relación lineal
    )
    direccion = "positiva" if r >= 0 else "negativa"  # ¿suben juntas (positiva) o una sube y la otra baja (negativa)?
    return f"{fuerza} y {direccion}"  # frase final, p. ej. "fuerte y negativa"


def _recta(columna_x: str, columna_y: str) -> tuple[float, float, float, "np.ndarray", "np.ndarray"]:
    """Ajusta y = m*x + b con numpy y devuelve (m, b, r2, x, y)."""
    df = cargar_dataset()  # DataFrame con caché
    x = df[columna_x].to_numpy()  # convierte la columna X a array de numpy (lo exige polyfit)
    y = df[columna_y].to_numpy()  # convierte la columna Y a array de numpy
    pendiente, interseccion = np.polyfit(x, y, 1)  # ajuste de grado 1 (recta): devuelve pendiente m e intersección b
    y_pred = pendiente * x + interseccion  # valores que "diría" la recta para cada x
    r2 = float(1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())  # R² = 1 - SSE/SST: % de varianza que explica la recta (0..1)
    return float(pendiente), float(interseccion), r2, x, y  # empaqueta todo lo que necesitan las tools públicas


@tool  # herramienta pública: correlación de Pearson
def correlacion(columna_a: str, columna_b: str) -> str:
    """Calcula la correlación de Pearson entre dos columnas numéricas (valor entre -1 y 1)."""
    error = _validar_columnas(columna_a, columna_b)  # ¿alguna columna no es numérica?
    if error:  # si hubo error...
        return error  # ...lo devolvemos tal cual para que el LLM corrija su JSON
    df = cargar_dataset()  # DataFrame con caché
    r = float(df[columna_a].corr(df[columna_b]))  # corr() de pandas = correlación de Pearson entre ambas columnas
    return (
        f"Correlación de Pearson entre '{columna_a}' y '{columna_b}': r = {r:.3f}\n"  # el número r con 3 decimales
        f"Interpretación: relación {_interpretar(r)}."  # traducción a palabras (fuerza y dirección)
    )


@tool  # herramienta pública: recta de regresión
def recta_regresion(columna_x: str, columna_y: str) -> str:
    """Ajusta la recta y = m*x + b entre dos columnas numéricas y devuelve pendiente (m), intersección (b) y R²."""
    error = _validar_columnas(columna_x, columna_y)  # validación de columnas
    if error:
        return error  # reintento del LLM con nombres válidos
    pendiente, interseccion, r2, _, _ = _recta(columna_x, columna_y)  # calcula la recta (los _ descartan los arrays x,y que aquí no hacen falta)
    return (
        f"Recta ajustada: {columna_y} = {pendiente:.4f} * {columna_x} + {interseccion:.4f}\n"  # ecuación completa y = m·x + b
        f"- pendiente (m): {pendiente:.4f} -> por cada unidad extra de '{columna_x}', '{columna_y}' cambia ~{pendiente:.4f}\n"  # significado práctico de m
        f"- intersección (b): {interseccion:.4f}\n"  # valor de y cuando x = 0
        f"- R²: {r2:.4f} (explica el {r2:.1%} de la variabilidad)"  # bondad del ajuste en porcentaje
    )


@tool  # herramienta pública: predicción puntual con la recta
def predecir_con_regresion(columna_x: str, columna_y: str, valor_x: float) -> str:
    """Predice el valor de columna_y para un valor concreto de columna_x usando la recta de regresión ajustada."""
    error = _validar_columnas(columna_x, columna_y)  # validación de columnas
    if error:
        return error  # reintento del LLM
    pendiente, interseccion, _, _, _ = _recta(columna_x, columna_y)  # obtenemos m y b (r2 y arrays no hacen falta)
    prediccion = pendiente * valor_x + interseccion  # aplicamos la fórmula de la recta: y = m·x + b
    return (
        f"Predicción: si {columna_x} = {valor_x}, entonces {columna_y} ~ {prediccion:.3f}\n"  # resultado de la predicción
        f"(recta: {columna_y} = {pendiente:.4f} * {columna_x} + {interseccion:.4f})"  # transparencia: con qué recta se calculó
    )
