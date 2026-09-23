# HealthScroller — Documentación técnica del proyecto

Material de referencia para la presentación: qué es el proyecto, cómo está
construido, cómo funciona el sistema de agentes y para qué sirve cada
librería. Se centra en el proyecto, el sistema de agentes, Python y sus
librerías.

---

## 1. ¿Qué es HealthScroller?

**Problema.** Las redes sociales se usan cada vez más y a edades más
tempranas. ¿Cómo afecta eso al sueño, a la salud mental y al rendimiento
académico de los estudiantes?

**Datos.** Dataset `data/Social_media_impact_on_life.csv` con **4500
estudiantes**:

| Tipo | Columnas |
|---|---|
| Numéricas | `Age`, `Daily_Usage_Hours`, `Weekend_Extra_Hours`, `Sleep_Duration_Hours`, `Sleep_Quality_Score`, `Perceived_Stress_Score`, `Mental_Health_Index`, `Academic_Performance_GPA` |
| Categóricas | `Gender`, `Academic_Level`, `Primary_Platform`, `Device_Type`, `Late_Night_Usage`, `Social_Comparison_Frequency`, `Overall_Impact` |

**Solución (dos piezas que se conectan):**

1. **Análisis de datos con un notebook Jupyter** (`notebooks/clase2_health_scroller_analisis.ipynb`):
   limpia, visualiza, calcula estadísticas, correlaciones, regresión y
   simulaciones... y **exporta un resumen** (`outputs/analysis_results.json`).
2. **Un chatbot multi-agente en la terminal** que responde preguntas en
   lenguaje natural sobre esos datos, corriendo **100% en local** con el
   modelo de lenguaje `gemma3:1b` vía Ollama (gratis, sin APIs de pago).

**Ideas centrales del diseño:**

- **Los números salen siempre de pandas/numpy**, jamás de la imaginación
  del LLM (`gemma3:1b` es demasiado pequeño para hacer matemáticas fiables).
- El notebook es el **puente**: su exportación JSON (el *briefing*) se
  inyecta en el prompt de cada agente.
- `gemma3:1b` **no soporta *function calling*** nativo, así que el proyecto
  implementa su propio bucle de herramientas en JSON.

---

## 2. Arquitectura general

```text
   pregunta del usuario
           │
           ▼
┌─────────────────────┐   el orquestador responde SOLO una etiqueta
│  NODO ORQUESTADOR   │   (plan A: LLM · plan B: palabras clave)
│  orchestrator.py    │
└─────────┬───────────┘
          │  ruta: conversacion | estadistico | algebraico | simulacion
    ┌─────┴──────┬────────────┬──────────────┐
    ▼            ▼            ▼              ▼
┌──────────┐┌────────────┐┌────────────┐┌────────────┐
│conversac.││estadístico ││ algebraico ││ simulación │
│0 tools   ││ 3 tools    ││ 3 tools    ││ 3 tools    │
└────┬─────┘└─────┬──────┘└─────┬──────┘└─────┬──────┘
     │            └──────┬──────┴─────────────┘
     │                   ▼
     │        ┌───────────────────────────┐
     │        │ EXECUTOR (executor.py)    │  bucle de hasta 4 pasos:
     │        │ clase AgenteHerramientas  │  LLM → JSON → ejecuta tool →
     │        │                           │  resultado → LLM → texto final
     │        └─────────────┬─────────────┘
     │                      ▼
     │        ┌───────────────────────────┐
     │        │ 9 HERRAMIENTAS (@tool)    │────► data/*.csv (en vivo)
     │        │ pandas + numpy            │      4500 estudiantes
     │        └───────────────────────────┘
     ▼
respuesta final — la CLI la muestra como [ruta]
```

**Flujo de datos del briefing** (la otra fuente de conocimiento):

```text
notebook (sección 7) ──exporta──► outputs/analysis_results.json
                                        │
                    knowledge.py lo lee (cargar_briefing)
                                        │
                    briefing_texto() se inyecta en TODOS los prompts
```

Tres frases para recordar la arquitectura:

1. El **orquestador** no responde: solo clasifica y deriva.
2. Cada **especialista** es un LLM + un conjunto de **herramientas** que
   ejecuta Python real sobre el CSV.
3. El **briefing JSON** da contexto al modelo; las **tools** dan los
   números exactos.

---

## 3. Estructura del proyecto

```text
HealthScroller/
├── data/
│   └── Social_media_impact_on_life.csv     dataset (4500 estudiantes)
├── notebooks/
│   ├── clase1_ml_precio_autos.ipynb        clase 1 (regresión de precios)
│   └── clase2_health_scroller_analisis.ipynb   análisis + exporta el briefing
├── outputs/
│   └── analysis_results.json               generado por el notebook (local)
├── src/health_scroller/                    código del chatbot (paquete Python)
│   ├── config.py                           rutas + parámetros de Ollama
│   ├── knowledge.py                        carga del CSV y del briefing (con caché)
│   ├── tools/                              las 9 herramientas @tool
│   │   ├── stats_tools.py                  medias, conteos, resumen
│   │   ├── algebra_tools.py                correlación, recta, predicción
│   │   └── simulation_tools.py             Monte Carlo, bootstrap, what-if
│   ├── agents/                             el sistema de agentes
│   │   ├── orchestrator.py                 clasificador (etiquetas)
│   │   ├── executor.py                     bucle de herramientas JSON
│   │   ├── conversation.py                 agente conversacional (sin tools)
│   │   ├── statistics.py                   agente estadístico
│   │   ├── algebra.py                      agente algebraico
│   │   └── simulation.py                   agente de simulación
│   ├── graph.py                            grafo LangGraph (orquestador → especialistas)
│   └── cli.py                              chat en la terminal
├── requirements.txt                        las 9 librerías (se explican abajo)
├── pyproject.toml                          define el paquete y el comando `healthscroller`
└── .env.example                            ejemplo de configuración opcional
```

**Capas del `src/`** (de abajo arriba):

| Capa | Ficheros | Responsabilidad |
|---|---|---|
| Configuración | `config.py` | rutas, modelo, temperatura |
| Datos | `knowledge.py` | leer CSV y briefing una sola vez (caché) |
| Herramientas | `tools/*.py` | cálculo real con pandas/numpy |
| Agentes | `agents/*.py` | prompts + orquestación + bucle de tools |
| Orquestación | `graph.py` | quién responde a quién (LangGraph) |
| Interfaz | `cli.py` | hablar con el sistema desde la terminal |

**Punto de entrada de la CLI:** `python -m health_scroller.cli`
(también disponible como comando `healthscroller`, definido en
`pyproject.toml`). La CLI muestra los mensajes como `[etiqueta] mensaje`,
y permite forzar un agente con `/agente <nombre>`.

---

## 4. El sistema de agentes

### 4.1. Flujo completo de una pregunta

1. El usuario escribe una pregunta en la CLI.
2. El **nodo orquestador** extrae la última pregunta del historial y pide
   al LLM que responda **SOLO con una etiqueta**.
3. La etiqueta se valida contra la lista `ETIQUETAS`; si el LLM falla, hay
   un **plan B por palabras clave**.
4. LangGraph enruta la pregunta al **nodo especialista** correspondiente.
5. El especialista (clase `AgenteHerramientas`) ejecuta su **bucle de
   herramientas**: pide un JSON, ejecuta la tool con Python, devuelve el
   resultado al modelo... hasta que el modelo redacta la respuesta final.
6. La CLI imprime `[ruta] respuesta`.

### 4.2. El orquestador (`agents/orchestrator.py`)

No es un "chatbot": es un **clasificador** con doble trampa pensada para
un modelo pequeño:

| Mecanismo | Cómo funciona |
|---|---|
| **Plan A (LLM)** | Prompt que pide clasificar en UNA etiqueta; respuesta con `temperature=0.0` (máxima determinismidad) |
| **Validación** | Se recorre `ETIQUETAS` **en orden de prioridad** (de lo más específico a lo más general) y se comprueba si la etiqueta aparece en la respuesta |
| **Plan B (palabras clave)** | `clasificar_por_palabras()`: normaliza mayúsculas y acentos (`unicodedata`) y busca palabras de cada etiqueta (`PALABRAS_CLAVE`) |
| **Red de seguridad** | Si Ollama ni siquiera responde (`except`), se usa directamente el plan B; por defecto cae en `conversacion` |

**Las etiquetas actuales:** `simulacion`, `algebraico`, `estadistico`,
`conversacion` (el orden del tuple es la prioridad).

**Componentes configurables del orquestador:**

- `ETIQUETAS` — qué rutas existen en el grafo.
- `PROMPT_ORQUESTADOR` — descripción de cada etiqueta + ejemplos few-shot
  (`"¿Cuál es la media de GPA por género?" -> estadistico`).
- `PALABRAS_CLAVE` — red de seguridad sin LLM.

### 4.3. El executor (`agents/executor.py`): el corazón técnico

**Por qué existe.** Los modelos de Ollama *grandes* devuelven llamadas a
funciones estructuradas; `gemma3:1b` no. La solución es un **bucle ReAct
en texto plano** implementado con la clase `AgenteHerramientas`:

**a) Construcción del prompt.** Al crear el agente:

1. Se guarda el prompt del especialista (con el briefing).
2. `construir_seccion_herramientas()` genera una sección `## Herramientas
   disponibles` leyendo **el esquema JSON de cada tool**
   (`tool.tool_call_schema`): nombre, parámetros con su tipo, si tienen
   valor por defecto o son opcionales, y su descripción (el docstring).
3. Formato que debe devolver el modelo:
   `{"herramienta": "<nombre>", "argumentos": {"<param>": "<valor>"}}`.

**b) El bucle** (máximo `MAX_PASOS = 4` iteraciones):

```text
llamo al LLM ──► ¿respondió un JSON con herramienta?
                     │ no                          │ sí
                     ▼                             ▼
            respuesta final               ejecuto la tool en Python
            (texto normal)                        │
                     ▲              ¿nombre desconocido o error?
                     │                    │ error            │ ok
                     │                    ▼                  ▼
                     │         mensaje de error con    resultado de
                     │         la firma EXACTA: el     pandas/numpy
                     │         modelo debe reintentar       │
                     └──────── resultado inyectado como ◄───┘
                              mensaje de usuario + "responde
                              ya con texto normal (sin JSON)"
```

**Detalles de robustez:**

- `extraer_llamada_json()`: localiza el primer `{`, cuenta llaves
  balanceadas ignorando comillas y escapados, y hace `json.loads`;
  tolera que el modelo envuelva el JSON en un bloque de código marcado
  como `json`.
- Claves tolerantes: acepta `herramienta`/`tool`/`nombre`/`name` y
  `argumentos`/`args`/`arguments`.
- `_preparar_argumentos()`: convierte `"3.5"` (texto) a `float`/`int`
  según el esquema de la tool.
- Si se agotan los 4 pasos: mensaje pidiendo reformular la pregunta.
- Interfaz `invoke(state) -> {"messages": [...]}`: mismo contrato que un
  nodo de LangGraph, por eso puede colgarse directamente del grafo.

### 4.4. Los cuatro especialistas

| Agente | Etiqueta | Módulo / creador | Herramientas | Misión |
|---|---|---|---|---|
| Conversación | `conversacion` | `conversation.py` → `crear_agente_conversacion()` | **ninguna** (`tools=[]`) | Saludos, guía del proyecto, deriva al especialista adecuado |
| Estadístico | `estadistico` | `statistics.py` → `crear_agente_estadistico()` | `resumen_estadistico`, `media_por_grupo`, `conteo_categorias` | Medias, medianas, conteos, comparaciones entre grupos |
| Algebraico | `algebraico` | `algebra.py` → `crear_agente_algebraico()` | `correlacion`, `recta_regresion`, `predecir_con_regresion` | Correlación de Pearson, recta `y = m·x + b`, predicciones |
| Simulación | `simulacion` | `simulation.py` → `crear_agente_simulacion()` | `monte_carlo_media`, `bootstrap_intervalo_confianza`, `escenario_what_if` | Monte Carlo, intervalos de confianza, escenarios "¿qué pasaría si...?" |

**Estructura común de TODO agente** (patrón que se repite en los 4
módulos):

```python
PROMPT_XXX = f"""Eres el Chatbot ... de HealthScroller.

Datos clave (briefing del notebook):
{briefing_texto()}                      # 1) contexto precalculado

Tienes herramientas para ...             # 2) qué puede hacer
Reglas:
- Usa SIEMPRE una herramienta ...       # 3) reglas duras (nunca inventar)
- Argumentos EXACTOS ...
Ejemplo de llamada correcta:
{{"herramienta": "...", "argumentos": {{...}}}}   # 4) ejemplo few-shot
                                                   #    (llaves dobles {{ }})


def crear_agente_xxx():
    return AgenteHerramientas(           # 5) fábrica: prompt + tools
        prompt=PROMPT_XXX,
        tools=[...],
    )
```

### 4.5. Cómo se crea y se configura un agente nuevo

**Crear** (5 pasos):

1. **Módulo nuevo** en `src/health_scroller/agents/<nombre>.py`.
2. **Prompt** con las 4 piezas: briefing (`briefing_texto()`), reglas,
   y ejemplo JSON con doble llave.
3. **Fábrica** `crear_agente_<nombre>()` que devuelve
   `AgenteHerramientas(prompt=..., tools=[...])`.
4. **Registro en `graph.py`**: import + línea en el diccionario
   `agentes = {...}`. El resto del grafo (nodos, aristas, rutas
   condicionales) se genera solo a partir de ese diccionario.
5. **Alta en `orchestrator.py`**: añadir la etiqueta a `ETIQUETAS`,
   una línea + un ejemplo en `PROMPT_ORQUESTADOR`, y palabras clave en
   `PALABRAS_CLAVE`.

**Configuración — qué se controla desde dónde:**

| Qué | Dónde | Valor actual |
|---|---|---|
| Personalidad y reglas del agente | su `PROMPT_XXX` (módulo del agente) | por agente |
| Qué cálculos puede hacer | lista `tools=[...]` de su fábrica | por agente |
| Temperatura del LLM | `config.py` → `TEMPERATURA` | `0.2` (creatividad baja) |
| Temperatura del orquestador | `orchestrator.py` (inline) | `0.0` (clasificación seca) |
| Ventana de contexto | `config.py` → `NUM_CTX` | `8192` (cabe el briefing) |
| Modelo y servidor | `config.py` → `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | `gemma3:1b` / `localhost:11434` |
| Máximo de pasos de tool | `executor.py` → `MAX_PASOS` | `4` |

### 4.6. El grafo (`graph.py`)

- `EstadoChat(MessagesState)` = historial de mensajes + campo `ruta`.
- `nodo_orquestador` extrae la última pregunta y llama a
  `clasificar_intencion()`.
- `_enrutar` devuelve `estado["ruta"]`; las aristas condicionales usan
  `list(agentes.keys())` (añadir una clave al dict añade la ruta).
- `crear_grafo()` devuelve **(grafo compilado, diccionario de agentes)**:
  el grafo para enrutar y el diccionario para que la CLI pueda invocar
  agentes directamente con `/agente <nombre>`.

---

## 5. El sistema de tools (herramientas)

### 5.1. Qué es una tool para el LLM

Una tool es una **función Python con nombre, descripción y esquema de
argumentos** que el modelo puede "pedir" ejecutar. El decorador
`@tool` (de `langchain_core`) genera automáticamente:

- `tool.name` — el nombre que escribe el modelo en el JSON.
- `tool.description` — el **docstring**: lo único que el modelo lee para
  decidir cuándo usarla.
- `tool.tool_call_schema` — los argumentos con su tipo, que el executor
  inyecta en el prompt y usa para convertir tipos.

### 5.2. Anatomía de una tool del proyecto

```python
@tool
def media_por_grupo(columna_grupo: str, columna_valor: str) -> str:
    """Calcula la media de columna_valor para cada grupo (docstring = contrato)."""
    df = cargar_dataset()                    # datos con caché (knowledge.py)
    if columna_grupo not in COLUMNAS_CATEGORICAS:   # validación amable
        return f"Error: '{columna_grupo}' no es categórica. Usa una de: ..."
    medias = df.groupby(columna_grupo)[columna_valor].mean().round(2)
    ...                                       # cálculo real (pandas/numpy)
    return f"Media de '{columna_valor}' por '{columna_grupo}':\n..."   # SIEMPRE str
```

**Convenciones que siguen las 9 tools:**

1. **Devuelven siempre `str`** (texto legible, nunca `dict` ni `DataFrame`).
2. **Validan las columnas** contra las listas `COLUMNAS_NUMERICAS` /
   `COLUMNAS_CATEGORICAS` y devuelven un **error en español**: el executor
   lo reenvía al modelo para que reintente con los nombres exactos.
3. **Nunca calculan "a ojo"**: todo pasa por `cargar_dataset()` (que además
   imputa los nulos con la mediana, igual que el notebook).
4. El **docstring describe la semántica**, no la implementación.

### 5.3. Catálogo completo (9 tools)

**`tools/stats_tools.py` — estadística descriptiva (pandas)**

| Tool | Firma | Qué calcula |
|---|---|---|
| `resumen_estadistico` | `(columna)` | media, mediana, desv. típica, mín/máx, Q1/Q3 de una columna numérica |
| `media_por_grupo` | `(columna_grupo, columna_valor)` | `groupby` + `mean`: media del valor por cada categoría (ej.: GPA por plataforma) |
| `conteo_categorias` | `(columna)` | `value_counts` con porcentaje por categoría |

**`tools/algebra_tools.py` — álgebra (numpy + pandas)**

| Tool | Firma | Qué calcula |
|---|---|---|
| `correlacion` | `(columna_a, columna_b)` | correlación de Pearson `r` + interpretación en palabras (`_interpretar`: débil/moderada/fuerte/muy fuerte, positiva/negativa) |
| `recta_regresion` | `(columna_x, columna_y)` | `np.polyfit(x, y, 1)` → pendiente `m`, intersección `b` y R² (varianza explicada) |
| `predecir_con_regresion` | `(columna_x, columna_y, valor_x)` | aplica `y = m·x + b` para un valor concreto de `x` |

Función auxiliar compartida `_recta()`: convierte a arrays con
`to_numpy()`, ajusta con `polyfit` y calcula R² a mano con la fórmula
`1 - SSE/SST`.

**`tools/simulation_tools.py` — simulación (numpy)**

| Tool | Firma | Qué calcula |
|---|---|---|
| `monte_carlo_media` | `(columna, n_simulaciones=1000, tamano_muestra=500)` | muchas muestras aleatorias → cómo se comporta la media |
| `bootstrap_intervalo_confianza` | `(columna, n_repeticiones=2000)` | remuestreo *con* reemplazo → IC 95% con `np.percentile([2.5, 97.5])` |
| `escenario_what_if` | `(columna_x, columna_y, valor_actual, valor_nuevo)` | proyecta con la recta de regresión qué pasaría al cambiar `x` de un valor a otro |

**Reproducibilidad:** un único `RNG = np.random.default_rng(42)` (semilla
fija) para que las simulaciones de clase sean repetibles.

### 5.4. Cómo se crea una tool nueva (receta)

1. Función con `@tool` en el fichero de tools correspondiente (o uno
   nuevo dentro de `src/health_scroller/tools/`).
2. Docstring claro en español (es lo que lee el modelo).
3. Argumentos con tipos simples (`str`, `float`, `int`) y **devolución
   `str`**.
4. Dentro: `cargar_dataset()` + validación + cálculo + texto formateado.
5. Importarla en el módulo del agente y añadirla a su lista `tools=[...]`.
   El executor la incluye sola en el prompt (lee el esquema
   automáticamente): **no hay que registrarla en ningún otro sitio**.

---

## 6. Configuración del sistema (`config.py`)

Fichero único que **lee primero** todo el paquete:

```python
load_dotenv()   # python-dotenv: si existe un .env, sus variables mandan

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "gemma3:1b")
TEMPERATURA     = float(os.getenv("TEMPERATURA", "0.2"))
NUM_CTX         = 8192          # ventana de contexto de los prompts
DATA_PATH       = ... / "data" / "Social_media_impact_on_life.csv"
RESULTS_PATH    = ... / "outputs" / "analysis_results.json"
```

| Parámetro | Papel | Efecto práctico |
|---|---|---|
| `OLLAMA_BASE_URL` | dónde vive el servidor Ollama | por defecto, la propia máquina |
| `OLLAMA_MODEL` | modelo cargado | `gemma3:1b`: pequeño, rápido, gratuito |
| `TEMPERATURA` | aleatoriedad de las respuestas | `0.2` = respuestas serias y repetibles |
| `NUM_CTX` | tokens máximos del prompt | `8192` para que quepa el briefing completo |
| `DATA_PATH` / `RESULTS_PATH` | rutas a los datos y al briefing | calculadas desde la raíz del proyecto |

Las variables del `.env` (ejemplo en `.env.example`) permiten cambiar
modelo o temperatura **sin tocar código**; si no hay `.env`, valen los
valores por defecto de arriba.

---

## 7. Tecnologías y librerías

Las 9 dependencias de `requirements.txt`, en sus tres bloques. Cada una:
qué es, por qué esa versión y **cómo se usa en este proyecto**.

### 7.1. Bloque A — Análisis de datos

#### pandas `>=2.2,<3`

**Qué es.** La librería de datos por excelencia de Python: su objeto
`DataFrame` es una tabla (como Excel/SQL) con las que se hacen casi todos
los cálculos.

**Cómo la usamos aquí:**

| Sitio | Uso concreto |
|---|---|
| `knowledge.py` | `pd.read_csv()` del dataset; imputación de nulos con `.fillna(...median())` en `Perceived_Stress_Score` y `Academic_Performance_GPA` |
| `stats_tools.py` | `.groupby().mean()`, `.value_counts()`, `.mean()`, `.median()`, `.std()`, `.quantile()`, `.sort_values()`, `.round()` |
| `algebra_tools.py` | `.corr()` de Pearson y `.to_numpy()` para pasarlo a numpy |
| `simulation_tools.py` / `executor` | acceso por columna y conversión a arrays |
| notebook | carga, EDA, métricas, correlaciones: todo el análisis de la clase |

#### numpy `>=1.26,<3`

**Qué es.** Arrays numéricos de alto rendimiento y toda la matemática
"seria" (álgebra, estadística, azar controlado).

**Cómo la usamos aquí:**

| Sitio | Uso concreto |
|---|---|
| `algebra_tools.py` | `np.polyfit(x, y, 1)` → recta de regresión; aritmética de arrays para R² |
| `simulation_tools.py` | `np.random.default_rng(42)` (semilla fija), `RNG.choice()` para muestrear, `np.array` de miles de medias, `.mean()/.std()`, `np.percentile(..., [2.5, 97.5])` para el IC 95% |
| notebook | mismos cálculos en las secciones 5 y 6 + estilos numéricos |

#### matplotlib `>=3.8,<4`

**Qué es.** El motor de gráficas "bajo nivel" de Python
(`matplotlib.pyplot`): figuras, ejes, líneas, subplots.

**Cómo la usamos aquí** (solo en los notebooks; el chatbot no dibuja):

- `fig, axes = plt.subplots(2, 4, ...)` para paneles de histogramas,
  barras y boxplots.
- `plt.plot(x_linea, pendiente * x_linea + interseccion, color="red")`
  → la recta de regresión superpuesta al scatter (sección 5.3).
- `plt.title/label/tight_layout/show` para etiquetar y mostrar.

#### seaborn `>=0.13,<0.14`

**Qué es.** Visualización **encima** de matplotlib, con estética
estadística lista para usar (tipos de gráficos con un par de líneas).

**Cómo la usamos aquí** (notebook):

| Función | Gráfico |
|---|---|
| `sns.histplot(..., kde=True)` | distribuciones de las variables numéricas con curva de densidad |
| `sns.countplot(...)` | frecuencia de las variables categóricas |
| `sns.boxplot(..., hue=..., palette="pastel")` | GPA/salud/estrés según el impacto declarado |
| `sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)` | mapa de correlaciones |
| `sns.scatterplot(..., alpha=0.25)` | nube de puntos de la regresión |

### 7.2. Bloque B — Notebook de clase

#### notebook `>=7.2`

**Qué es.** El paquete que aporta **Jupyter Notebook**: un servidor local
con un **kernel** de Python que ejecuta celdas de código y texto en el
navegador, guardando código + resultados + gráficas en un `.ipynb`.

**Cómo lo usamos aquí:** `notebooks/clase2_health_scroller_analisis.ipynb`
es el documento vivo de la clase, con estas secciones:

1. Problema a resolver · 2. Carga del dataset · 3. EDA (limpieza de
nulos, visualización numéricas/categóricas, métricas clave,
correlaciones) · 4. Estadística descriptiva → **alimenta al chatbot
estadístico** · 5. Correlaciones y regresión → **alimenta al algebraico**
· 6. Simulación → **alimenta al de simulación** · 7. **Exportación: el
puente hacia los agentes** (genera `outputs/analysis_results.json`) ·
8. Probar el chatbot.

Por eso el notebook es **obligatorio antes de usar el chatbot**: es quien
genera el briefing JSON que todos los prompts necesitan.

### 7.3. Bloque C — Agentes (LangChain + LangGraph + Ollama)

#### langchain `>=0.3,<0.4`

**Qué es.** El framework de referencia para aplicaciones con LLMs: define
los conceptos comunes (mensajes, prompts, herramientas, modelos).

**Cómo lo usamos aquí** (en concreto, su núcleo `langchain_core`):

- **Mensajes:** `SystemMessage` (el prompt del agente),
  `HumanMessage` (la pregunta... y también los *resultados* de las tools
  devueltos al modelo), `AIMessage` (lo que redacta el LLM).
- **`@tool`**: convierte una función Python en herramienta con nombre,
  docstring y esquema — es la base de las 9 tools.
- **`tool.tool_call_schema`**: de ahí sale la sección de herramientas
  del prompt y la conversión de tipos del executor.
- Del paquete principal (`langchain`) usamos el ecosistema como
  dependencia de coherencia; el *agente ReAct* prehecho
  (`create_react_agent`) **no** se usa: `gemma3:1b` no tiene
  tool-calling, así que el bucle es el nuestro (`executor.py`).

#### langchain-ollama `>=0.3,<0.4`

**Qué es.** El adaptador que conecta los objetos de LangChain con el
servidor **Ollama** (LLMs locales).

**Cómo lo usamos aquí:** la clase **`ChatOllama`** aparece en dos sitios:

| Sitio | Configuración | Papel |
|---|---|---|
| `executor.py` (`AgenteHerramientas`) | `model`, `base_url`, `temperature=TEMPERATURA` (0.2), `num_ctx` | el "cerebro" de cada especialista |
| `orchestrator.py` | `temperature=0.0` | clasificación de etiquetas lo más estable posible |

El modelo detrás es **`gemma3:1b`** (~1B parámetros): rápido en un
portátil, gratuito, y pequeño... por eso todo lo numérico recae en las
tools. (Ollama es el programa que lo sirve; no es una librería de Python,
es el servidor al que apunta `base_url`.)

#### langgraph `>=0.3,<0.6`

**Qué es.** Biblioteca de **grafos de estado** para flujos con LLMs
(Nodos = funciones; Aristas = transiciones; Estado = dato compartido).

**Cómo la usamos aquí** (`graph.py`, 60 líneas):

```python
from langgraph.graph import END, START, MessagesState, StateGraph

class EstadoChat(MessagesState):    # estado: lista de mensajes + campo "ruta"
    ruta: str

builder = StateGraph(EstadoChat)
builder.add_node("orquestador", nodo_orquestador)
for nombre, agente in agentes.items():          # 4 especialistas
    builder.add_node(nombre, agente)

builder.add_edge(START, "orquestador")                     # entrada
builder.add_conditional_edges("orquestador", _enrutar,     # bifurcación
                              list(agentes.keys()))
for nombre in agentes:
    builder.add_edge(nombre, END)                          # salida

grafo = builder.compile()
```

- `START`/`END`: nodos especiales de entrada y salida.
- `add_conditional_edges`: el orquestador decide el siguiente nodo según
  `estado["ruta"]`.
- `builder.compile()`: grafo listo para `grafo.invoke({"messages": [...]})`.
- **Extensibilidad:** al ser los nodos y las rutas un `dict`, añadir un
  agente es añadir una clave.

#### python-dotenv `>=1.0,<2`

**Qué es.** Carga variables de entorno desde un fichero `.env`
(clave=valor) al ejecutar el programa.

**Cómo lo usamos aquí:** una línea en `config.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

A partir de ahí, `os.getenv("OLLAMA_MODEL", "gemma3:1b")` respeta lo del
`.env` si existe, o los valores por defecto si no. Separa **configuración
local** de **código**: cambiar de modelo o temperatura no exige editar
`.py`. El fichero `.env.example` documenta las claves disponibles
(`OLLAMA_BASE_URL`, `OLLAMA_MODEL`).

### 7.4. Resumen para la presentación

| Librería | Papel en una frase | Dónde manda |
|---|---|---|
| pandas | tablas y estadística descriptiva | tools de stats, knowledge, notebook |
| numpy | matemática, regresión y azar controlado | tools de álgebra y simulación |
| matplotlib | figuras y ejes | notebook |
| seaborn | gráficos estadísticos bonitos | notebook |
| notebook | entorno interactivo de la clase + exporta el briefing | `notebooks/` |
| langchain | mensajes y `@tool` (el contrato LLM↔Python) | tools, executor, prompts |
| langchain-ollama | `ChatOllama`: puente con el modelo local | executor, orquestador |
| langgraph | el grafo: orquestador → especialista → fin | `graph.py` |
| python-dotenv | configuración por `.env` sin tocar código | `config.py` |

---

*HealthScroller — proyecto educativo. Documento técnico de apoyo a la
presentación.*
