"""Herramientas de simulación del Chatbot de Simulación (numpy + pandas)."""

import numpy as np
from langchain_core.tools import tool

from ..knowledge import cargar_dataset
from .stats_tools import COLUMNAS_NUMERICAS

# Semilla fija: los experimentos son reproducibles (buena práctica educativa)
RNG = np.random.default_rng(42)


def _validar(columna: str) -> str | None:
    if columna not in COLUMNAS_NUMERICAS:
        return f"Error: '{columna}' no es numérica. Usa: {', '.join(COLUMNAS_NUMERICAS)}"
    return None


@tool
def monte_carlo_media(columna: str, n_simulaciones: int = 1000, tamano_muestra: int = 500) -> str:
    """Simulación Monte Carlo: extrae muchas muestras aleatorias de una columna y estudia cómo se comporta su media."""
    error = _validar(columna)
    if error:
        return error
    datos = cargar_dataset()[columna].to_numpy()
    medias = np.array(
        [RNG.choice(datos, size=tamano_muestra).mean() for _ in range(n_simulaciones)]
    )
    return (
        f"Monte Carlo sobre '{columna}':\n"
        f"- simulaciones: {n_simulaciones} muestras de {tamano_muestra} estudiantes\n"
        f"- media de las medias: {medias.mean():.3f}\n"
        f"- desviación de las medias: {medias.std():.3f}\n"
        f"- media mínima / máxima observada: {medias.min():.3f} / {medias.max():.3f}"
    )


@tool
def bootstrap_intervalo_confianza(columna: str, n_repeticiones: int = 2000) -> str:
    """Calcula por bootstrap el intervalo de confianza al 95% de la media de una columna."""
    error = _validar(columna)
    if error:
        return error
    datos = cargar_dataset()[columna].to_numpy()
    medias = np.array(
        [RNG.choice(datos, size=datos.size, replace=True).mean() for _ in range(n_repeticiones)]
    )
    ic_bajo, ic_alto = np.percentile(medias, [2.5, 97.5])
    return (
        f"Bootstrap 95% para la media de '{columna}' ({n_repeticiones} repeticiones):\n"
        f"- intervalo de confianza: [{ic_bajo:.3f}, {ic_alto:.3f}]\n"
        f"- media real del dataset: {datos.mean():.3f}"
    )


@tool
def escenario_what_if(columna_x: str, columna_y: str, valor_actual: float, valor_nuevo: float) -> str:
    """Escenario '¿qué pasaría si...?': con la recta de regresión estima cuánto cambiaría columna_y al pasar columna_x de valor_actual a valor_nuevo."""
    error_x, error_y = _validar(columna_x), _validar(columna_y)
    if error_x or error_y:
        return error_x or error_y
    df = cargar_dataset()
    x = df[columna_x].to_numpy()
    y = df[columna_y].to_numpy()
    pendiente, interseccion = np.polyfit(x, y, 1)
    y_actual = pendiente * valor_actual + interseccion
    y_nuevo = pendiente * valor_nuevo + interseccion
    cambio = y_nuevo - y_actual
    return (
        f"Escenario: pasar '{columna_x}' de {valor_actual} a {valor_nuevo}.\n"
        f"- {columna_y} estimado actual: {y_actual:.3f}\n"
        f"- {columna_y} estimado nuevo: {y_nuevo:.3f}\n"
        f"- cambio estimado: {cambio:+.3f}\n"
        f"(usando la recta {columna_y} = {pendiente:.4f} * {columna_x} + {interseccion:.4f})"
    )
