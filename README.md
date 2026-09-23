# HealthScroller

Proyecto educativo (Clase 2) sobre el impacto de las redes sociales en la vida
de los estudiantes. Combina **análisis de datos con Pandas/Numpy** (notebook) y
un **chatbot multi-agente con LangGraph** que responde preguntas usando un
modelo local de **Ollama (`gemma3:1b`)**: gratis, sin APIs de pago y corriendo
en tu portátil.

## Arquitectura

```
                        ┌────────────────────┐
   pregunta del usuario │  AGENTE ORQUESTADOR │  clasifica la intención
          ─────────────►│  (LangGraph + LLM)  │
                        └─────────┬──────────┘
                                  │
        ┌─────────────┬───────────┼───────────────┐
        ▼             ▼           ▼               ▼
  ┌───────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
  │conversación│ │estadístico │ │ algebraico │ │ simulación │
  │  (saludos, │ │ (medias,   │ │ (correlac.,│ │ (Monte     │
  │  guía)     │ │  groupby)  │ │  regresión)│ │  Carlo,    │
  └───────────┘ └─────┬──────┘ └─────┬──────┘ │  bootstrap,│
                      │              │        │  what-if)  │
                      ▼              ▼        └─────┬──────┘
              ┌────────────────────────────────────────┐
              │  HERRAMIENTAS (pandas / numpy reales)  │
              │  consultan data/*.csv en vivo          │
              └────────────────┬───────────────────────┘
                               │
              ┌────────────────▼───────────────────────┐
              │  outputs/analysis_results.json        │
              │  (briefing generado por el notebook)   │
              └────────────────────────────────────────┘
```

Ideas clave:

- **Cada número que ves en el chat sale de pandas/numpy**, nunca de la
  imaginación del LLM (`gemma3:1b` es muy pequeño para hacer matemáticas).
- El notebook exporta un **briefing JSON** que se inyecta en los *prompts*.
- `gemma3:1b` no soporta *function calling* nativo de Ollama, así que el
  proyecto implementa un bucle de herramientas en JSON (`agents/executor.py`):
  el modelo pide `{"herramienta": ..., "argumentos": ...}`, Python ejecuta y
  devuelve el resultado.

## Estructura del proyecto

```
HealthScroller/
├── data/                                  dataset de los estudiantes
│   └── Social_media_impact_on_life.csv
├── notebooks/
│   ├── clase1_ml_precio_autos.ipynb       clase 1 (regresión de precios)
│   └── clase2_health_scroller_analisis.ipynb   <-- EJECUTA ESTE (clase 2)
├── outputs/
│   └── analysis_results.json              briefing que alimenta a los agentes
├── src/health_scroller/
│   ├── config.py                          rutas + configuración de Ollama
│   ├── knowledge.py                       carga del CSV y del briefing
│   ├── tools/                             herramientas pandas/numpy
│   │   ├── stats_tools.py                 medias, conteos, groupby
│   │   ├── algebra_tools.py               correlaciones, regresión
│   │   └── simulation_tools.py            Monte Carlo, bootstrap, what-if
│   ├── agents/
│   │   ├── orchestrator.py                decide qué agente responde
│   │   ├── executor.py                    bucle de herramientas (JSON)
│   │   ├── conversation.py                chatbot de conversación
│   │   ├── statistics.py                  chatbot estadístico
│   │   ├── algebra.py                     chatbot algebraico
│   │   └── simulation.py                  chatbot de simulación
│   ├── graph.py                           grafo de LangGraph
│   └── cli.py                             interfaz de terminal
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Requisitos

- Python 3.11 o superior
- [Ollama](https://ollama.com) instalado en tu equipo
- (Opcional) Docker Desktop para ejecutar el chatbot en contenedor

## Puesta en marcha

### 1. Modelo de Ollama (solo la primera vez)

```bash
ollama pull gemma3:1b
```

### 2. Entorno virtual y dependencias

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3. Ejecuta el notebook de análisis (genera el briefing)

```bash
jupyter notebook notebooks/clase2_health_scroller_analisis.ipynb
```

Ejecuta todas las celdas (Kernel > Restart & Run All). Al final se crea
`outputs/analysis_results.json`, el archivo que nutre a los agentes.

### 4. Arranca el chatbot

Asegúrate de que Ollama está corriendo (`ollama serve` o el icono de la app) y:

```bash
python -m health_scroller.cli
# o, equivalentemente:
healthscroller
```

Ejemplos de preguntas:

| Pregunta | Agente |
|---|---|
| `Hola, ¿quién eres?` | conversación |
| `¿Cuál es la media del GPA por género?` | estadístico |
| `¿Qué correlación hay entre horas de uso y sueño?` | algebraico |
| `¿Qué pasaría si reduzco el uso a 3 horas?` | simulación |

Comandos de la CLI:

- `/ayuda` — muestra la ayuda
- `/agente <nombre>` — fuerza a un agente (`estadistico`, `algebraico`,
  `simulacion`, `conversacion`)
- `/reset` — borra el historial
- `/salir` — cierra el chat

### 5. (Opcional) Ejecutar con Docker

El contenedor usa el Ollama de tu equipo (debe estar arrancado):

```bash
docker compose up --build
```

## Configuración

Copia `.env.example` a `.env` si necesitas cambiar algo:

```bash
OLLAMA_BASE_URL=http://localhost:11434   # URL de Ollama
OLLAMA_MODEL=gemma3:1b                   # modelo a usar
```

## Tecnologías

- **Pandas / Numpy / Matplotlib / Seaborn** — análisis de datos
- **LangChain / LangGraph** — agentes y grafo de enrutado
- **Ollama** — LLM local gratuito (`gemma3:1b`)
- **Jupyter Notebook** — análisis de la clase
- **Docker** — empaquetado del sistema
