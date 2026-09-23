"""Herramientas estadísticas del Chatbot Estadístico (todo con pandas).

Cada función se decora con @tool: así LangChain la convierte en una
herramienta que el LLM puede elegir y llamar por su nombre.
"""

from langchain_core.tools import tool

from ..knowledge import cargar_dataset

COLUMNAS_NUMERICAS = [
    "Age",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
    "Academic_Performance_GPA",
]

COLUMNAS_CATEGORICAS = [
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Device_Type",
    "Late_Night_Usage",
    "Social_Comparison_Frequency",
    "Overall_Impact",
]


@tool
def resumen_estadistico(columna: str) -> str:
    """Calcula media, mediana, desviación típica, mínimo, máximo y cuartiles de una columna numérica del dataset."""
    df = cargar_dataset()
    if columna not in COLUMNAS_NUMERICAS:
        return f"Error: '{columna}' no es numérica. Usa una de: {', '.join(COLUMNAS_NUMERICAS)}"
    serie = df[columna]
    return (
        f"Resumen de '{columna}' ({serie.size} estudiantes):\n"
        f"- media: {serie.mean():.2f}\n"
        f"- mediana: {serie.median():.2f}\n"
        f"- desviación típica: {serie.std():.2f}\n"
        f"- mínimo: {serie.min():.2f} | máximo: {serie.max():.2f}\n"
        f"- Q1: {serie.quantile(0.25):.2f} | Q3: {serie.quantile(0.75):.2f}"
    )


@tool
def media_por_grupo(columna_grupo: str, columna_valor: str) -> str:
    """Calcula la media de columna_valor para cada grupo de columna_grupo (ej.: media del GPA por Gender)."""
    df = cargar_dataset()
    if columna_grupo not in COLUMNAS_CATEGORICAS:
        return f"Error: '{columna_grupo}' no es categórica. Usa una de: {', '.join(COLUMNAS_CATEGORICAS)}"
    if columna_valor not in COLUMNAS_NUMERICAS:
        return f"Error: '{columna_valor}' no es numérica. Usa una de: {', '.join(COLUMNAS_NUMERICAS)}"
    medias = df.groupby(columna_grupo)[columna_valor].mean().round(2).sort_values(ascending=False)
    lineas = [f"- {grupo}: {media:.2f}" for grupo, media in medias.items()]
    return f"Media de '{columna_valor}' por '{columna_grupo}':\n" + "\n".join(lineas)


@tool
def conteo_categorias(columna: str) -> str:
    """Cuenta cuántos estudiantes hay en cada categoría de una columna categórica (con porcentaje)."""
    df = cargar_dataset()
    if columna not in COLUMNAS_CATEGORICAS:
        return f"Error: '{columna}' no es categórica. Usa una de: {', '.join(COLUMNAS_CATEGORICAS)}"
    conteos = df[columna].value_counts()
    lineas = [
        f"- {categoria}: {conteo} ({conteo / df.shape[0]:.1%})"
        for categoria, conteo in conteos.items()
    ]
    return f"Conteo de '{columna}':\n" + "\n".join(lineas)
