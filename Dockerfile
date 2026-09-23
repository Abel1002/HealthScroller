# Imagen base ligera de Python
FROM python:3.11-slim

WORKDIR /app

# 1) Instalar dependencias primero (mejora la caché de Docker)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copiar código y recursos
COPY src ./src
COPY data ./data
COPY outputs ./outputs

# 3) Configuración: el contenedor habla con el Ollama del host
ENV PYTHONPATH=/app/src
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=gemma3:1b

CMD ["python", "-m", "health_scroller.cli"]
