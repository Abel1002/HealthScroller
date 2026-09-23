"""Herramientas de simulación del Chatbot de Simulación (numpy + pandas).

QUÉ HACE:
    Expone 3 herramientas (@tool) de muestreo: Monte Carlo, bootstrap
    (intervalos de confianza) y escenarios "¿qué pasaría si...?".

PARA QUÉ SIRVE:
    Son la "calculadora" del agente de simulación: responden a preguntas de
    probabilidad con datos reales, no con intuiciones.

CUÁNDO SE EJECUTA:
    Solo cuando el bucle del executor detecta un JSON con su nombre
    (p. ej. {"herramienta": "monte_carlo_media", "argumentos": {...}}).
    Pueden tardar unos segundos (miles de muestreos).
"""

import numpy as np  # numpy: generación aleatoria y estadística de arrays (la base de estas simulaciones)
from langchain_core.tools import tool  # decorador que convierte la función en herramienta para el LLM

from ..knowledge import cargar_dataset  # DataFrame limpio con caché (.. = un nivel arriba)
from .stats_tools import COLUMNAS_NUMERICAS  # lista blanca reutilizada de stats_tools

# Semilla fija: los experimentos son reproducibles (buena práctica educativa)
# default_rng(42) = generador de números aleatorios CON semilla: cada ejecución
# produce EXACTAMENTE los mismos "aleatorios" (clave para corregir la clase)
RNG = np.random.default_rng(42)


def _validar(columna: str) -> str | None:
    # Auxiliar interna (no es tool): comprueba que la columna sea numérica
    if columna not in COLUMNAS_NUMERICAS:  # ¿está en la lista blanca?
        return f"Error: '{columna}' no es numérica. Usa: {', '.join(COLUMNAS_NUMERICAS)}"  # error que el executor reenvía al LLM
    return None  # None = validación superada


@tool  # herramienta pública: simulación Monte Carlo
def monte_carlo_media(columna: str, n_simulaciones: int = 1000, tamano_muestra: int = 500) -> str:
    """Simulación Monte Carlo: extrae muchas muestras aleatorias de una columna y estudia cómo se comporta su media."""
    error = _validar(columna)  # ¿columna numérica?
    if error:
        return error  # reintento del LLM con nombre válido
    datos = cargar_dataset()[columna].to_numpy()  # extrae la columna como array de numpy (4500 valores)
    medias = np.array(
        [RNG.choice(datos, size=tamano_muestra).mean() for _ in range(n_simulaciones)]  # n_simulaciones muestras de tamaño 500 (CON reemplazo) y su media
    )
    return (
        f"Monte Carlo sobre '{columna}':\n"  # cabecera del informe
        f"- simulaciones: {n_simulaciones} muestras de {tamano_muestra} estudiantes\n"  # parámetros del experimento
        f"- media de las medias: {medias.mean():.3f}\n"  # centralidad de las medias muestrales
        f"- desviación de las medias: {medias.std():.3f}\n"  # cuánto varían entre sí (error estándar de la media)
        f"- media mínima / máxima observada: {medias.min():.3f} / {medias.max():.3f}"  # rango de lo observado en las simulaciones
    )


@tool  # herramienta pública: intervalo de confianza por bootstrap
def bootstrap_intervalo_confianza(columna: str, n_repeticiones: int = 2000) -> str:
    """Calcula por bootstrap el intervalo de confianza al 95% de la media de una columna."""
    error = _validar(columna)  # validación de la columna
    if error:
        return error  # reintento del LLM
    datos = cargar_dataset()[columna].to_numpy()  # array con los 4500 valores reales
    medias = np.array(
        [RNG.choice(datos, size=datos.size, replace=True).mean() for _ in range(n_repeticiones)]  # remuestreo CON reemplazo (mismo tamaño) x2000: es el bootstrap
    )
    ic_bajo, ic_alto = np.percentile(medias, [2.5, 97.5])  # percentiles 2.5 y 97.5 = IC 95% (deja el 2.5% fuera a cada lado)
    return (
        f"Bootstrap 95% para la media de '{columna}' ({n_repeticiones} repeticiones):\n"  # cabecera con los parámetros
        f"- intervalo de confianza: [{ic_bajo:.3f}, {ic_alto:.3f}]\n"  # rango en el que cae la media "verdadera" con 95% de confianza
        f"- media real del dataset: {datos.mean():.3f}"  # contraste: la media de todos los datos
    )


@tool  # herramienta pública: escenario contrafactual ("¿qué pasaría si...?")
def escenario_what_if(columna_x: str, columna_y: str, valor_actual: float, valor_nuevo: float) -> str:
    """Escenario '¿qué pasaría si...?': con la recta de regresión estima cuánto cambiaría columna_y al pasar columna_x de valor_actual a valor_nuevo."""
    error_x, error_y = _validar(columna_x), _validar(columna_y)  # valida las DOS columnas involucradas
    if error_x or error_y:
        return error_x or error_y  # devuelve el primer error encontrado (si lo hay)
    df = cargar_dataset()  # DataFrame con caché
    x = df[columna_x].to_numpy()  # columna X como array de numpy
    y = df[columna_y].to_numpy()  # columna Y como array de numpy
    pendiente, interseccion = np.polyfit(x, y, 1)  # ajuste de la recta y = m·x + b (grado 1)
    y_actual = pendiente * valor_actual + interseccion  # valor de y HOY (con el valor actual de x)
    y_nuevo = pendiente * valor_nuevo + interseccion  # valor de y en el ESCENARIO propuesto
    cambio = y_nuevo - y_actual  # diferencia: cuánto sube o baja y (con signo + / -)
    return (
        f"Escenario: pasar '{columna_x}' de {valor_actual} a {valor_nuevo}.\n"  # enunciado del contrafáctico
        f"- {columna_y} estimado actual: {y_actual:.3f}\n"  # predicción con los datos de hoy
        f"- {columna_y} estimado nuevo: {y_nuevo:.3f}\n"  # predicción bajo el escenario
        f"- cambio estimado: {cambio:+.3f}\n"  # delta con signo explícito (+sube / -baja)
        f"(usando la recta {columna_y} = {pendiente:.4f} * {columna_x} + {interseccion:.4f})"  # transparencia del modelo usado
    )
