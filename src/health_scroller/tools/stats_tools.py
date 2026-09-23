"""Herramientas estadísticas del Chatbot Estadístico (todo con pandas).

QUÉ HACE:
    Expone 3 herramientas (@tool) que el LLM puede llamar para calcular
    estadística descriptiva REAL sobre el dataset: resumen de una columna,
    medias por grupo y conteo de categorías.

PARA QUÉ SIRVE:
    Son la "calculadora" del agente estadístico: sin ellas el modelo
    inventaría números; con ellas, cada respuesta sale de pandas.

CUÁNDO SE EJECUTA:
    NO se ejecutan al arrancar. Cada tool solo corre cuando el bucle del
    executor detecta que el modelo ha pedido un JSON con su nombre
    (p. ej. {"herramienta": "media_por_grupo", "argumentos": {...}}).
"""

from langchain_core.tools import tool  # decorador: convierte la función en objeto Tool (nombre + docstring + esquema)

from ..knowledge import cargar_dataset  # carga del CSV con caché (.. = subir un nivel al paquete health_scroller)

# Columnas del dataset que contienen NÚMEROS: solo ellas se pueden promediar o correlacionar
# Funciona de lista blanca: si el LLM escribe otra columna, la tool responde un error útil
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

# Columnas CATEGÓRICAS (texto con pocas opciones): válidas para AGRUPAR o CONTAR
COLUMNAS_CATEGORICAS = [
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Device_Type",
    "Late_Night_Usage",
    "Social_Comparison_Frequency",
    "Overall_Impact",
]


@tool  # registra la función como herramienta: su docstring es lo que lee el LLM para elegirla
def resumen_estadistico(columna: str) -> str:
    """Calcula media, mediana, desviación típica, mínimo, máximo y cuartiles de una columna numérica del dataset."""
    df = cargar_dataset()  # pide el DataFrame limpio (1ª vez lee el CSV; después sale de la caché)
    if columna not in COLUMNAS_NUMERICAS:  # validación: ¿pidió una columna numérica?
        return f"Error: '{columna}' no es numérica. Usa una de: {', '.join(COLUMNAS_NUMERICAS)}"  # el error se devuelve al LLM para que reintente con nombres válidos
    serie = df[columna]  # extrae solo esa columna (una Serie de pandas con 4500 valores)
    return (  # informe en texto plano (las tools SIEMPRE devuelven str, es su contrato)
        f"Resumen de '{columna}' ({serie.size} estudiantes):\n"  # cabecera con el nº de filas
        f"- media: {serie.mean():.2f}\n"  # promedio aritmético con 2 decimales
        f"- mediana: {serie.median():.2f}\n"  # valor central de la distribución (resistente a valores extremos)
        f"- desviación típica: {serie.std():.2f}\n"  # cuánto se dispersan los datos respecto a la media
        f"- mínimo: {serie.min():.2f} | máximo: {serie.max():.2f}\n"  # extremos realies del rango
        f"- Q1: {serie.quantile(0.25):.2f} | Q3: {serie.quantile(0.75):.2f}"  # cuartiles: 25% y 75% de los datos
    )


@tool  # misma conversión a herramienta que la anterior
def media_por_grupo(columna_grupo: str, columna_valor: str) -> str:
    """Calcula la media de columna_valor para cada grupo de columna_grupo (ej.: media del GPA por Gender)."""
    df = cargar_dataset()  # DataFrame con caché
    if columna_grupo not in COLUMNAS_CATEGORICAS:  # el campo por el que se AGRUPA debe ser categórico
        return f"Error: '{columna_grupo}' no es categórica. Usa una de: {', '.join(COLUMNAS_CATEGORICAS)}"  # mensaje de reintento para el LLM
    if columna_valor not in COLUMNAS_NUMERICAS:  # lo que se PROMEDIA debe ser numérico
        return f"Error: '{columna_valor}' no es numérica. Usa una de: {', '.join(COLUMNAS_NUMERICAS)}"  # mensaje de reintento para el LLM
    medias = df.groupby(columna_grupo)[columna_valor].mean().round(2).sort_values(ascending=False)  # agrupa -> media -> 2 decimales -> ordena de mayor a menor
    lineas = [f"- {grupo}: {media:.2f}" for grupo, media in medias.items()]  # compone 1 línea de texto por grupo (comprehension de lista)
    return f"Media de '{columna_valor}' por '{columna_grupo}':\n" + "\n".join(lineas)  # cabecera + líneas unidas con saltos de línea


@tool  # conversión a herramienta
def conteo_categorias(columna: str) -> str:
    """Cuenta cuántos estudiantes hay en cada categoría de una columna categórica (con porcentaje)."""
    df = cargar_dataset()  # DataFrame con caché
    if columna not in COLUMNAS_CATEGORICAS:  # solo se pueden contar columnas de la lista blanca
        return f"Error: '{columna}' no es categórica. Usa una de: {', '.join(COLUMNAS_CATEGORICAS)}"  # reintento del LLM
    conteos = df[columna].value_counts()  # frecuencia absoluta de cada categoría, de más a menos frecuente
    lineas = [
        f"- {categoria}: {conteo} ({conteo / df.shape[0]:.1%})"  # recuento + % sobre el total (df.shape[0] = 4500 filas)
        for categoria, conteo in conteos.items()  # recorre cada categoría y su número de apariciones
    ]
    return f"Conteo de '{columna}':\n" + "\n".join(lineas)  # cabecera + una línea por categoría
