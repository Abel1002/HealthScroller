"""Herramientas algebraicas del Chatbot Algebraico (numpy + pandas)."""

import numpy as np
from langchain_core.tools import tool

from ..knowledge import cargar_dataset
from .stats_tools import COLUMNAS_NUMERICAS


def _validar_columnas(columna_a: str, columna_b: str) -> str | None:
    """Devuelve un mensaje de error si alguna columna no es numérica."""
    malas = [c for c in (columna_a, columna_b) if c not in COLUMNAS_NUMERICAS]
    if malas:
        return f"Error: {', '.join(malas)} no son numéricas. Usa: {', '.join(COLUMNAS_NUMERICAS)}"
    return None


def _interpretar(r: float) -> str:
    """Traduce el valor de r a palabras (fuerza y dirección)."""
    fuerza = (
        "muy fuerte" if abs(r) > 0.7
        else "fuerte" if abs(r) > 0.5
        else "moderada" if abs(r) > 0.3
        else "débil"
    )
    direccion = "positiva" if r >= 0 else "negativa"
    return f"{fuerza} y {direccion}"


def _recta(columna_x: str, columna_y: str) -> tuple[float, float, float, "np.ndarray", "np.ndarray"]:
    """Ajusta y = m*x + b con numpy y devuelve (m, b, r2, x, y)."""
    df = cargar_dataset()
    x = df[columna_x].to_numpy()
    y = df[columna_y].to_numpy()
    pendiente, interseccion = np.polyfit(x, y, 1)
    y_pred = pendiente * x + interseccion
    r2 = float(1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(pendiente), float(interseccion), r2, x, y


@tool
def correlacion(columna_a: str, columna_b: str) -> str:
    """Calcula la correlación de Pearson entre dos columnas numéricas (valor entre -1 y 1)."""
    error = _validar_columnas(columna_a, columna_b)
    if error:
        return error
    df = cargar_dataset()
    r = float(df[columna_a].corr(df[columna_b]))
    return (
        f"Correlación de Pearson entre '{columna_a}' y '{columna_b}': r = {r:.3f}\n"
        f"Interpretación: relación {_interpretar(r)}."
    )


@tool
def recta_regresion(columna_x: str, columna_y: str) -> str:
    """Ajusta la recta y = m*x + b entre dos columnas numéricas y devuelve pendiente (m), intersección (b) y R²."""
    error = _validar_columnas(columna_x, columna_y)
    if error:
        return error
    pendiente, interseccion, r2, _, _ = _recta(columna_x, columna_y)
    return (
        f"Recta ajustada: {columna_y} = {pendiente:.4f} * {columna_x} + {interseccion:.4f}\n"
        f"- pendiente (m): {pendiente:.4f} -> por cada unidad extra de '{columna_x}', '{columna_y}' cambia ~{pendiente:.4f}\n"
        f"- intersección (b): {interseccion:.4f}\n"
        f"- R²: {r2:.4f} (explica el {r2:.1%} de la variabilidad)"
    )


@tool
def predecir_con_regresion(columna_x: str, columna_y: str, valor_x: float) -> str:
    """Predice el valor de columna_y para un valor concreto de columna_x usando la recta de regresión ajustada."""
    error = _validar_columnas(columna_x, columna_y)
    if error:
        return error
    pendiente, interseccion, _, _, _ = _recta(columna_x, columna_y)
    prediccion = pendiente * valor_x + interseccion
    return (
        f"Predicción: si {columna_x} = {valor_x}, entonces {columna_y} ~ {prediccion:.3f}\n"
        f"(recta: {columna_y} = {pendiente:.4f} * {columna_x} + {interseccion:.4f})"
    )
