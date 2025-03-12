# Usamos la imagen base de Jupyter que ya trae un entorno Python listo para notebooks
FROM jupyter/base-notebook:python-3.9

# Instalamos dependencias del sistema necesarias (por ejemplo, compiladores)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Regresamos al usuario no privilegiado de la imagen base
USER $NB_UID

# Copiamos el archivo de requerimientos y lo instalamos
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Descargamos los modelos de spaCy necesarios
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download de_core_news_sm

# Exponemos el puerto para JupyterLab
EXPOSE 8888

# CMD sin deshabilitar la autenticación por token, se usará la configuración segura por defecto.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

