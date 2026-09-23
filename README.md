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

### Concepto: ¿qué es un entorno virtual?

Antes de ejecutar los comandos del paso 8, conviene entender qué hacen.

Un **entorno virtual** (la carpeta `.venv`) es una **copia aislada de Python
solo para este proyecto**: su propio intérprete (`python.exe`) y su propia
carpeta de librerías (`site-packages`), sin tocar tu instalación global de
Python.

**Para qué sirve:**

- **Aislar dependencias:** cada proyecto o clase lleva sus librerías sin
  romper las de otros proyectos ni la instalación global de Python.
- **Reproducibilidad:** todo el grupo instala las mismas versiones (las fija
  `requirements.txt`): el clásico *"en mi máquina funciona"* deja de ser un
  problema.
- **Instalación limpia:** si algo se estropea, borras `.venv` y lo vuelves a
  crear; tu Python global no se entera de nada.

**Cómo funciona:**

```text
sin activar:  terminal ──► C:\Python311\python.exe               (Python global)
                              (p. ej.; la ruta de tu instalación)

activado:     terminal ──► HealthScroller\.venv\Scripts\python.exe
              (.venv)       pandas, numpy, langgraph... se instalan AQUÍ dentro
```

- `python -m venv .venv` **crea** la carpeta con su Python de dentro.
- `.venv\Scripts\Activate.ps1` **activa** el entorno: durante esa ventana de
  terminal, los comandos `python` y `pip` apuntan al `.venv` (el prompt
  muestra `(.venv)`). Para salir del entorno: `deactivate`.
- **No se sube a GitHub:** `.venv` está en `.gitignore` porque pesa cientos
  de MB y cada equipo lo crea en su propio equipo (paso 8).
- Elegir el "intérprete" en VS Code es decirle al editor **qué** Python usa.

*Analogía:* es como la mochila de herramientas de un trabajo concreto: te
llevas solo lo necesario, no mezclas material de otros trabajos y si la
pierdes, te haces otra igual a partir de la lista de la compra
(`requirements.txt`).

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
repositorio**, se genera en tu equipo.

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

## Tu propio agente

¿Ya funciona el chatbot? Ahora toca el reto de la clase: **crear tu propio
agente** que responda una **pregunta nueva** sobre los datos y publicar tu
versión del proyecto en **tu** cuenta de GitHub.

El ejemplo guiado completo es el **Agente Plataformas** (etiqueta
`plataformas`), que responde a:

> ¿Qué red social se asocia a peor GPA?

Sigue los pasos A-G **en orden**. Si tu pregunta es diferente, el
procedimiento es idéntico: solo cambian el nombre, el prompt y (si hace
falta) las herramientas.

### A. Elige tu pregunta y el nombre de tu agente

- **Nombre** en minúsculas y **sin acentos ni espacios** (`plataformas`,
  `riesgo`, `sueno`...): será la clave del grafo, la etiqueta del orquestador
  y lo que escribirás después en `/agente <nombre>`.
- Escribe tu pregunta tipo en **una frase concreta** (ej.: *«¿Qué red social
  se asocia a peor GPA?»*).
- Piensa qué herramientas necesitas: ¿con las ya existentes basta
  (`resumen_estadistico`, `media_por_grupo`, `conteo_categorias`,
  correlaciones, what-if...) o necesitas un cálculo nuevo (paso E)?

Ideas de preguntas si no se te ocurre ninguna:

| Pregunta | Herramientas |
|---|---|
| ¿Qué red social se asocia a peor GPA? *(el ejemplo)* | `media_por_grupo`, `conteo_categorias` |
| ¿Quién pasa más horas en el móvil: chicos o chicas? | `media_por_grupo` |
| ¿Los que se comparan poco tienen mejor salud mental? | `media_por_grupo`, `conteo_categorias` |
| ¿Qué correlación hay entre estrés y horas de sueño? | `tools/algebra_tools.py` |
| ¿Qué pasaría con el GPA si durmieran 8 horas? | `tools/simulation_tools.py` |

### B. Crea el módulo de tu agente (archivo nuevo)

Crea `src/health_scroller/agents/plataformas.py` copiando el patrón de
`statistics.py`:

```python
"""Agente Plataformas: compara el impacto de cada red social."""

from ..knowledge import briefing_texto
from ..tools.stats_tools import conteo_categorias, media_por_grupo
from .executor import AgenteHerramientas

PROMPT_PLATAFORMAS = f"""Eres el Agente de Plataformas de HealthScroller.

Datos clave (briefing del notebook):
{briefing_texto()}

Tu misión: responder preguntas del tipo "¿Qué red social se asocia a peor GPA?"
o "¿Cuántos estudiantes usan cada plataforma?".

Tienes herramientas para consultar el dataset real de 4500 estudiantes.
Reglas:
- Usa SIEMPRE una herramienta para calcular; nunca inventes números.
- Los nombres de los argumentos deben coincidir EXACTAMENTE con el del listado de herramientas.
- Responde en español, claro y ordenado, citando los números que devuelve la herramienta.
- Termina con una interpretación breve de lo que significan esos números.

Ejemplo de llamada correcta para "¿Qué red social se asocia a peor GPA?":
{{"herramienta": "media_por_grupo", "argumentos": {{"columna_grupo": "Primary_Platform", "columna_valor": "Academic_Performance_GPA"}}}}"""


def crear_agente_plataformas():
    """Devuelve el agente de plataformas (con sus herramientas pandas)."""
    return AgenteHerramientas(
        prompt=PROMPT_PLATAFORMAS,
        tools=[media_por_grupo, conteo_categorias],
    )
```

Cuatro reglas al adaptarlo a **tu** agente:

1. El prompt debe ser un f-string (`f"""..."""`) con `briefing_texto()` para
   inyectar el resumen del notebook.
2. Dentro del f-string, el JSON de ejemplo lleva **doble llave** `{{` `}}`
   (Python se reserva `{` en las f-strings).
3. Regla sagrada en el prompt: **usar SIEMPRE las herramientas, nunca
   inventar números** (`gemma3:1b` es pequeño; las herramientas son su
   calculadora).
4. El ejemplo JSON del prompt es la pista principal que tiene el modelo para
   acertar: usa nombres de herramientas y argumentos **EXACTOS**.

### C. Regístralo en el grafo (modificar `graph.py`)

En `src/health_scroller/graph.py` solo hacen falta **dos cambios**:

1. Añade el import (junto a los demás):

```python
from .agents.plataformas import crear_agente_plataformas
```

2. Añade tu agente al diccionario `agentes`:

```python
agentes = {
    "conversacion": crear_agente_conversacion(),
    "estadistico": crear_agente_estadistico(),
    "algebraico": crear_agente_algebraico(),
    "simulacion": crear_agente_simulacion(),
    "plataformas": crear_agente_plataformas(),   # <-- NUEVO
}
```

**No toques nada más.** Los bucles de `crear_grafo()` añaden el nodo, el
borde hasta `END` y la ruta condicional **automáticamente** a partir de ese
diccionario; la CLI acepta `/agente plataformas` porque también lee
`agentes.keys()`.

*(Cosmético opcional: actualiza el `Literal` de `_enrutar`, el docstring y
el `BANNER` de `cli.py` para listar también tu agente.)*

### D. Enséñale al orquestador a detectarlo (modificar `orchestrator.py`)

En `src/health_scroller/agents/orchestrator.py`, **tres cambios**:

1. **`ETIQUETAS`**: añade tu etiqueta **al principio** (el orden es la
   prioridad; así no se la come `estadistico`):

```python
ETIQUETAS = ("plataformas", "simulacion", "algebraico", "estadistico", "conversacion")
```

2. **`PROMPT_ORQUESTADOR`**: añade una línea de etiqueta y un ejemplo de
   pregunta:

```text
- plataformas: comparaciones entre redes sociales (TikTok, Instagram...) o plataformas
...
"¿Qué red social se asocia a peor GPA?" -> plataformas
```

3. **`PALABRAS_CLAVE`** (el plan B, por si el LLM se equivoca): añade una
   entrada con las palabras de tu tema:

```python
"plataformas": ["plataforma", "red social", "redes sociales", "tiktok", "instagram"],
```

### E. (Opcional) Crea tu propia herramienta

Si tu pregunta necesita un cálculo que **no existe**, crea
`src/health_scroller/tools/tu_herramientas.py`:

```python
from langchain_core.tools import tool

from ..knowledge import cargar_dataset


@tool
def pct_estudiantes_sueno_insuficiente(horas_minimas: float) -> str:
    """Calcula el porcentaje de estudiantes que duermen menos de las horas indicadas."""
    df = cargar_dataset()
    pct = (df["Sleep_Duration_Hours"] < horas_minimas).mean()
    return f"El {pct:.1%} duerme menos de {horas_minimas} horas."
```

Luego impórtala en el módulo de tu agente y agrégala a la lista
`tools=[...]`. Reglas: la **descripción** es lo único que ve el LLM para
elegirla (escríbela en español y con claridad), los argumentos deben ser
tipos simples (`str`, `float`...), y devuelve **siempre** un `str`.

### F. Pruébalo

```powershell
python -m health_scroller.cli
```

1. Fuerza tu agente: `/agente plataformas` y escribe tu pregunta → la
   respuesta debe empezar por `[plataformas]`.
2. Sal del chat (`/salir`) y vuelve a entras **sin forzar**: escribe la misma
   pregunta y el orquestador debe mandarla a `[plataformas]`.

| Si falla | Revisa |
|---|---|
| `/agente plataformas` responde "Agentes disponibles: ..." | el import o la línea del diccionario en `graph.py` |
| El orquestador manda la pregunta a otro agente | `ETIQUETAS`, prompt y `PALABRAS_CLAVE` en `orchestrator.py` |
| Error `FileNotFoundError` al arrancar | falta el paso 9 (ejecutar el notebook) |
| El agente no usa la herramienta / inventa números | el ejemplo JSON del prompt y los nombres exactos de los argumentos |

### G. Publica tu proyecto en tu GitHub

Si has seguido los pasos anteriores, tu agente ya está probado en la carpeta
original. Ahora publica **todo** (incluido tu agente) en **tu** cuenta de
GitHub practicando el flujo completo de Git desde cero:

**1. Copia el proyecto sin el historial ni el `.venv`** (desde la carpeta que
contiene `HealthScroller`; `robocopy` copia todo salvo `.git` y `.venv`,
e ignora el resumen de estadísticas que imprime al terminar):

```powershell
robocopy HealthScroller HealthScroller-mio /E /XD .git .venv
cd HealthScroller-mio
```

**2. Inicializa un repositorio nuevo** (rama `main`) y dile a Git quién eres
(solo la primera vez en tu equipo):

```powershell
git init -b main
git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@ejemplo.com"
```

**3. Crea el repositorio vacío en tu perfil de GitHub:**

1. Entra en <https://github.com/new>.
2. Nombre: `HealthScroller` (o `HealthScroller-tu-nombre` para no
   confundirlo con el del profe).
3. **No marques** "Add a README file" ni el desplegable de `.gitignore`:
   el repo debe quedar **vacío**.
4. Pulsa **Create repository**. GitHub te mostrará unos comandos: son
   prácticamente los mismos que pegaremos ahora.

**4. Primer commit y push:**

```powershell
git add .
git commit -m "mi propio agente: plataformas"
git remote add origin https://github.com/TU_USUARIO/HealthScroller.git
git push -u origin main
```

**5. Comprueba:** abre `https://github.com/TU_USUARIO/HealthScroller` en el
navegador y verás todos tus archivos (incluido tu `plataformas.py`).
`outputs/analysis_results.json` **no se sube**: lo ignora `.gitignore` porque
se genera localmente con el notebook.

Notas del primer push:

- Git puede pedirte credenciales: entra con **tu** usuario de GitHub (aparece
  una ventana del Gestor de credenciales de Windows; se guarda para siempre).
- A partir de ahora, cada vez que cambies algo: `git add .` →
  `git commit -m "qué he hecho"` → `git push`.

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
