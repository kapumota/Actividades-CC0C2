### Cuadernos NLP 2025-2

* Cuaderno 1: [Librerías NLP](https://github.com/kapumota/Actividades-CC0C2/blob/main/2025-1/Librerias_NLP.ipynb)
* Cuaderno 2: [Cargador de datos](https://github.com/kapumota/Actividades-CC0C2/blob/main/2025-1/Cargador_datos_NLP.ipynb)
* Cuaderno 3: [Tokenización](https://github.com/kapumota/Actividades-CC0C2/blob/main/2025-1/Tokenizacion.ipynb)
* Cuaderno 4: [Algoritmo BPE](https://github.com/kapumota/Actividades-CC0C2/blob/main/2025-1/BPE.ipynb)
* Cuaderno 5: [Normalización, lemantización y segmentación de palabras](https://github.com/kapumota/Actividades-CC0C2/blob/main/2025-1/Normalizacion_lemantizacion.ipynb)


### Instrucciones 2025-1

#### 1. Preparar la carpeta de trabajo

1. Crea una carpeta (por ejemplo, `nlp-curso`) y coloca en ella:
   - El archivo `Dockerfile`.
   - El archivo `requirements.txt`.
   - Tus cuadernos Jupyter (`.ipynb`) que quieras usar en el curso (si ya los tienes preparados).

2. La estructura de tu carpeta podría ser:
   ```
   nlp-curso/
   ├── Dockerfile
   ├── requirements.txt
   └── notebooks/
       ├── notebook1.ipynb
       └── notebook2.ipynb
   ```
   *Los notebooks pueden estar en la misma carpeta o en una subcarpeta, según tu preferencia.*

#### 2. Construir la imagen Docker

##### 2.1 Desde Docker Desktop en Windows

1. Abre **Docker Desktop** y asegúrate de que esté corriendo correctamente.
2. Abre una terminal en Windows (símbolo del sistema o PowerShell).
3. Navega hasta la carpeta que contiene el `Dockerfile` y el `requirements.txt`. Por ejemplo:
   ```bash
   cd C:\ruta\a\nlp-curso
   ```
4. Ejecuta el comando de construcción de la imagen:
   ```bash
   docker build -t mi-imagen-nlp .
   ```
   Aquí, `-t mi-imagen-nlp` asigna el nombre (`mi-imagen-nlp`) a la imagen que se creará y el `.` indica el contexto de construcción actual (la carpeta donde está el Dockerfile).

##### 2.2 Desde línea de comandos en Linux

1. Asegúrate de tener Docker instalado y ejecutándose.
2. Abre una terminal y navega hasta la carpeta con tu `Dockerfile` y `requirements.txt`:
   ```bash
   cd /ruta/a/nlp-curso
   ```
3. Ejecuta:
   ```bash
   docker build -t mi-imagen-nlp .
   ```
   *Igual que en Windows, `-t` especifica el nombre de la imagen.*


#### 3. Ejecutar el contenedor y acceder a JupyterLab

##### 3.1 Indicaciones generales

1. Para que puedas editar y guardar los cuadernos desde tu máquina (y no sólo dentro del contenedor), es recomendable montar la carpeta de notebooks en el contenedor. Esto se hace con la opción `-v` (o `--volume`).
2. Expondremos el puerto 8888 (tal como se definió en el Dockerfile) y lo mapearemos al puerto 8888 de la máquina anfitriona. Esto se hace con la opción `-p 8888:8888`.

##### 3.2 Ejemplo de comando:

```bash
docker run -it --rm \
    -p 8888:8888 \
    -v /ruta/a/nlp-curso/notebooks:/home/jovyan/work \
    mi-imagen-nlp
```

- `-it`: modo interactivo con pseudo-TTY (para poder ver la salida y, si es necesario, entrar en bash).
- `--rm`: para eliminar el contenedor al salir (deja tu disco limpio).
- `-p 8888:8888`: mapea el puerto interno 8888 del contenedor al mismo puerto en tu máquina.
- `-v /ruta/a/nlp-curso/notebooks:/home/jovyan/work`: monta la carpeta local con los notebooks en la ruta `/home/jovyan/work` dentro del contenedor (que es donde JupyterLab, por defecto, ubica los archivos de trabajo).
   - En Windows (PowerShell), la ruta local podría verse así: `C:\ruta\a\nlp-curso\notebooks:/home/jovyan/work`
   - En Linux, algo como: `/home/usuario/nlp-curso/notebooks:/home/jovyan/work`

Al ejecutar este comando, verás en la terminal la salida de JupyterLab, que mostrará una URL con un token de acceso (por ejemplo, `http://127.0.0.1:8888/?token=...`). 

##### 3.3 Acceder a JupyterLab

1. Copia y pega la URL que se muestra en la terminal en tu navegador (por ejemplo, `http://127.0.0.1:8888/?token=<tu-token>`).
2. Verás el entorno de JupyterLab en tu navegador, con los cuadernos que tengas en la carpeta montada (`/home/jovyan/work`).


#### 4. Uso en Docker Desktop (Windows) de forma gráfica

Además de la línea de comandos, Docker Desktop en Windows también permite:

1. Ir a la pestaña **Images**.
2. Buscar la imagen que creaste (`mi-imagen-nlp`).
3. Hacer clic en **Run**.
4. Configurar los valores de puerto (8888) y el volumen (montar la carpeta de notebooks) en las opciones gráficas.
5. Hacer clic en **Run** para iniciar el contenedor.

Después, para acceder, repites el proceso de abrir la URL con el token en el navegador.

#### 5. Ajustes y consejos finales

- Si en lugar de `jupyter lab` prefieres usar `jupyter notebook`, puedes cambiar el comando final en el `Dockerfile`, o bien usar `jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser`.  
- Revisa que tu Dockerfile y `requirements.txt` estén en orden y que no haya conflictos de versiones.
- Si requieres instalar más paquetes del sistema (usando `apt-get`) o Python (usando `pip`), añádelos en el Dockerfile antes de exponer tu puerto para que queden en la imagen.


---
#### Proyectos del curso de procesamiento de lenguaje natural CC3S2 (2024-2)

##### Lista de proyectos

* Implementar desde cero mecanismos de atención dispersa (Sparse Attention), SliGLU, RMSNorm, MoE y Rope embedding en PyTorch.
* Ajustar finamente un LLM con PPO vs DPO vs ORPO utilizando el paquete PEFT.
* Entrenar un LLM de manera distribuida con el paquete Accelerate en AWS SageMaker utilizando la estrategia de optimización Zero Redundancy Optimizer.
* Construir un modelo autoregresivo que implemente una variante de atención secuencial con generación de texto paso a paso.
* Usar un LLM para generar imágenes a partir de descripciones textuales, integrando embeddings visuales y de texto.
* Optimizar el rendimiento del modelo mediante la búsqueda eficiente de hiperparámetros en un espacio complejo.
* Crear una aplicación interactiva de chat que utiliza GPT para responder en tiempo real, con soporte para WebSockets para comunicación continua.
* Entrenar y ajustar un LLM especializado en la clasificación de noticias por temas, usando técnicas de fine-tuning y transfer learning.
* Utilizar un Transformer entrenado en secuencias musicales para generar piezas musicales originales a partir de un tema o estilo dado.
* Usar embeddings contextuales para generar recomendaciones de películas basadas en descripciones de trama y preferencias del usuario.
* Desplegar una API de preguntas y respuestas basada en LLM finamente ajustado con búsqueda de documentos
* Utilizar Transformers para detectar anomalías en datos de series temporales, con aplicación en áreas como la monitorización de servidores o sistemas financieros.
* Optimizar un modelo BERT para dispositivos móviles con técnicas de pruning, quantization y knowledge distillation
* Implementar un GAN que pueda generar imágenes ajustando estilos específicos, usando un enfoque condicional en el generador.
* Crear una aplicación de resumen de documentos largos con un enfoque extractivo y abstractive en LLM
* Desarrollar un sistema de generación de diálogos no repetitivos en juegos usando GPT y RL (Reinforcement Learning)
* Crear un motor de búsqueda que pueda procesar consultas mixtas de texto e imágenes, devolviendo resultados relevantes de ambos tipos de medios.
* Ajustar finamente un LLM con datos médicos para tareas como clasificación de documentos clínicos o extracción de información específica.
* Crear un sistema de generación de texto condicional basado en estilos o temas específicos usando técnicas de control de generación en LLMs.
* Desplegar un servicio de generación automática de resúmenes de textos largos utilizando un LLM optimizado y microservicios en AWS Lambda.
* Ajustar finamente un modelo con QLoRA para aumentar el tamaño del contexto.
* Desplegar una API de aplicación LLM escalable con capacidades de streaming, KV-caching, batch continuo y generación de texto.
* Desplegar una aplicación RAG usando LangChain, FastAPI y LangServer
* Implementar desde cero mecanismos de atención multi-cabecera con variantes como la atención relacional y la atención local en PyTorch.
* Entrenar un modelo de lenguaje con técnicas de aprendizaje contrastivo para mejorar la comprensión de párrafos largos.
* Optimizar un modelo Transformer con técnicas de pruning y quantization para su despliegue en dispositivos edge.
* Desarrollar un sistema de recomendación utilizando modelos de lenguaje preentrenados en combinación con embeddings de usuarios y productos.
* Desplegar un chatbot de asistencia técnica utilizando un LLM finamente ajustado y Docker para su implementación en un entorno en la nube.
* Implementar un pipeline de procesamiento de lenguaje natural en tiempo real utilizando Apache Kafka y un modelo Transformer.
* Desarrollar un modelo de clasificación de sentimientos finamente ajustado con distilBERT en un entorno de producción utilizando TensorFlow Serving.
