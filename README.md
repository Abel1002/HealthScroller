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
│   └── .gitkeep                           analysis_results.json se genera
│                                          al ejecutar el notebook (no está
│                                          en el repositorio)
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
└── .env.example
```

---

# Guía de puesta en marcha

Sigue los pasos **en orden**. Los pasos 1 a 5 son instalaciones de programas
que solo haces **una vez** en tu portátil.

## Instalación previa (solo la primera vez)

### Paso 1. Instala Visual Studio Code

1. Descarga el instalador desde <https://code.visualstudio.com/>.
2. Ejecútalo y acepta los valores por defecto (botón **Next** hasta el final).
3. Al terminar, ábrelo. Verás una ventana vacía con el menú lateral.

### Paso 2. Instala las extensiones de Python y Jupyter

1. En VS Code, pulsa el icono de **extensiones** en la barra lateral
   izquierda (los cuatro cuadritos) o el atajo **Ctrl + Shift + X**.
2. En el buscador escribe `Python` y pulsa **Install** en la extensión
   **Python** (la de **Microsoft**, es la primera de la lista).
3. Busca ahora `Jupyter` e instala también la de **Microsoft**.
4. Si VS Code te pide reiniciar, hazlo.

### Paso 3. Instala Python (3.11 o superior)

1. Descarga el instalador desde <https://www/python.org/downloads/>.
2. **Muy importante (Windows):** en la primera pantalla de instalación
   marca la casilla **`Add python.exe to PATH`** antes de pulsar
   **Install Now**. Sin esta casilla, los comandos `python` no funcionarán
   en la terminal.
3. Al terminar, comprueba la instalación: en VS Code abre una terminal
   (**Ctrl + `** y escribe:

   ```powershell
   python --version
   ```

   Debe responder algo como `Python 3.11.x` (o superior).

### Paso 4. Instala Git

Git es necesario para descargar el repositorio (`git clone`) y para traer
actualizaciones (`git pull`).

1. Descarga desde <https://git-scm.com/download/win>.
2. Instala con los valores por defecto (sigue dando a **Next**).
3. Comprueba en la terminal de VS Code:

   ```powershell
   git --version
   ```

### Paso 5. Instala Ollama

Ollama es el programa que ejecuta el modelo de lenguaje **en local y gratis**.

1. Descarga desde <https://ollama.com/download> y elige tu sistema
   operativo (Windows).
2. Ejecuta el instalador con los valores por defecto.
3. Comprueba la instalación en la terminal:

   ```powershell
   ollama --version
   ```

4. Ollama queda arrancado en segundo plano (aparece un icono en la bandeja
   del sistema). Si no está, ábrelo desde el menú Inicio.

---

## Puesta en marcha del proyecto

### Paso 6. Abre la terminal de VS Code

1. Abre VS Code.
2. Pulsa **Ctrl + `** (tecla de acento, la de al lado de la P) o menú
   **Terminal > New Terminal**.
3. Aparece abajo una pestaña con un cursor `PS C:\...>`: es ahí donde
   pegas **todos** los comandos de los siguientes pasos.

### Paso 7. Descarga el proyecto (primera vez) o actualízalo

**Primera vez** (clonar el repositorio):

```powershell
git clone https://github.com/Abel1002/HealthScroller.git
cd HealthScroller
```

**Si ya lo tienes descargado** (traer los últimos cambios con `git pull`):

```powershell
cd HealthScroller
git pull
```

### Paso 8. Crea el entorno virtual e instala las dependencias

En la terminal (ya dentro de la carpeta `HealthScroller`) ejecuta los
comandos **uno a uno**:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Qué hace cada uno:

| Comando | Qué hace |
|---|---|
| `python -m venv .venv` | crea una carpeta aislada de Python para este proyecto |
| `.venv\Scripts\Activate.ps1` | "entra" en ese entorno (el prompt muestra `(.venv)`) |
| `pip install -r requirements.txt` | instala pandas, numpy, langgraph, jupyter... |
| `pip install -e .` | instala el paquete `health_scroller` (permite lanzar el chatbot) |

Notas importantes:

- VS Code puede mostrarte un aviso: *"Would you like to select a Python
  environment?"* → elige **Enter interpreter path > .venv > Scripts >
  python.exe**. Así Jupyter y VS Code usan el entorno del proyecto.
- **macOS / Linux:** en lugar de `.venv\Scripts\Activate.ps1` usa
  `source .venv/bin/activate`.
- Si la activación da un error de "scripts deshabilitados" en PowerShell,
  ejecuta este comando (solo afecta a la ventana actual) y repite la
  activación:

  ```powershell
  Set-ExecutionPolicy -Scope Process RemoteSigned
  ```

  *Alternativa sin activar:* usa siempre la ruta completa, por ejemplo
  `.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### Paso 9. Ejecuta el análisis de Jupyter (genera el JSON del chatbot)

Este paso es **obligatorio**: crea el archivo `outputs/analysis_results.json`
que el chatbot necesita para arrancar. El archivo **no viene en el
repositoritorio**, se genera en tu equipo.

**Opción recomendada (desde VS Code):**

1. En el Explorador de VS Code (icono de archivos, arriba a la izquierda),
   abre la carpeta `notebooks` y haz doble clic en
   `clase2_health_scroller_analisis.ipynb`.
2. Arriba a la derecha del notebook elige el kernel
   **Python 3 (.venv)** (el que acabamos de crear).
3. Menú superior: **Kernel > Restart Kernel and Run All**.
4. Espera a que se ejecuten todas las celdas (verás tablas y gráficas).
5. Comprueba en la terminal que se creó el archivo:

   ```powershell
   type outputs\analysis_results.json
   ```

**Opción alternativa (desde la terminal):**

```powershell
jupyter notebook notebooks/clase2_health_scroller_analisis.ipynb
```

Se abrirá el notebook en el navegador; ejecuta todas las celdas
(**Cell > Run All**).

> Si ejecutas el chatbot **sin** haber hecho este paso, te avisará con el
> mensaje: `Falta outputs/analysis_results.json` — vuelve al paso 9.

### Paso 10. Descarga el modelo de Ollama (solo la primera vez)

```powershell
ollama pull gemma3:1b
```

Tarda unos minutos (descarga ~800 MB). Comprueba que está con:

```powershell
ollama list
```

Debe aparecer una línea con `gemma3:1b`.

### Paso 11. Ejecuta el chatbot

```powershell
python -m health_scroller.cli
```

Verás el banner y el prompt `Tú >`. Escribe tu pregunta y pulsa Enter.
Para salir: `/salir`.

Ejemplos de preguntas:

| Pregunta | Agente que responde |
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

---

## Solución de problemas (paso 12)

| Problema | Solución |
|---|---|
| `'python' no se reconoce` | Reinstala Python marcando **Add python.exe to PATH** (paso 3) y reinicia VS Code |
| `'git' no se reconoce` | Reinstala Git (paso 4) y reinicia VS Code |
| `'ollama' no se reconoce` o el chat avisa que Ollama no responde | Abre la aplicación **Ollama** desde el menú Inicio (o ejecuta `ollama serve` en otra terminal) |
| Aviso: `el modelo 'gemma3:1b' no está descargado` | Ejecuta `ollama pull gemma3:1b` (paso 10) |
| Error: `Falta outputs/analysis_results.json` | Ejecuta el notebook completo (paso 9) y relanza el chatbot |
| `'Activate.ps1' no se puede ejecutar... políticas` | `Set-ExecutionPolicy -Scope Process RemoteSigned` y repite la activación (paso 8) |
| VS Code / Jupyter no encuentra `pandas` o usa otro Python | Selecciona el intérprete `.venv/Scripts/python.exe` (nota del paso 8) |
| Quiero traerme los últimos cambios del repositorio | `cd HealthScroller` y `git pull` (paso 7) |

## Configuración

Copia `.env.example` a `.env` solo si necesitas cambiar algo (por defecto
no hace falta):

```bash
OLLAMA_BASE_URL=http://localhost:11434   # URL de Ollama
OLLAMA_MODEL=gemma3:1b                   # modelo a usar
```

## Tecnologías

- **Pandas / Numpy / Matplotlib / Seaborn** — análisis de datos
- **LangChain / LangGraph** — agentes y grafo de enrutado
- **Ollama** — LLM local gratuito (`gemma3:1b`)
- **Jupyter Notebook** — análisis de la clase
- **Git / GitHub** — control de versiones
