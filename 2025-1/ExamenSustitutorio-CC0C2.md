## **Examen sustitutorio CC0C2**

**Fecha de presentación:** Martes 22 de julio 12:00.

### **Evaluación común (20 pts):**

* **8 pts**: Código y/o informe breve (objetivo, metodología, conclusiones). Además, debe incluir un **README claro y preciso** con instrucciones de instalación, ejecución y dependencias.

* **12 pts**: Video **mínimo 10 min** con audio claro explicando los conceptos, la ejecución y los resultados.

  * Videos de menos de 10 min puntúan como máximo **4 pts**.
  * En el video **deben mencionarse y relacionarse obligatoriamente con el proyecto** los siguientes términos clave, de lo contrario se aplicará una penalización de hasta **6 pts** en la evaluación:
  
  Transformer, modelado de lenguaje (language modeling),
  Tokenización, representaciones (embeddings), entrenamiento/ajuste de parámetros (training/fine-tuning), predicción, tamaño de contexto (context size), leyes de escalado (scaling laws),
  habilidades emergentes en LLMs, prompts, alucinaciones y sesgos en LLMs (hallucinations and biases in LLMs), control de la salida mediante ejemplos,
  arquitecturas y panorama de LLMs, introducción al prompting, LangChain y LlamaIndex, prompting con LangChain, generación aumentada por recuperación (retrieval-augmented generation),
  agentes, ajuste de parámetros (fine-tuning), despliegue (deployment).


> Los proyectos son individuales Las rúbricas del examen son las mismas de evaluaciones pasadas.

#### 1. Proyecto: Prompt‑Engineering interactivo con LangChain

**Objetivo**
Diseñar una aplicación interactiva (CLI o notebook) que permita experimentar con distintos estilos de prompt (zero‑, one‑, few‑shot) usando LangChain y LexiQuipu como base.

**Descripción**

1. Interfaz (CLI o notebook) para seleccionar:

   * Tipo de prompt (zero/one/few‑shot)
   * Hasta 3 ejemplos a incluir
   * Parámetros de generación (temperature, max\_tokens)
2. Cadena en LangChain que monte el prompt con ejemplos y llame al LLM.
3. Medición de:

   * Casos de hallucination (vs. respuestas "ideales").
   * Calidad cualitativa al variar temperature y el número de ejemplos.

**Pasos sugeridos**

1. Clona LexiQuipu y adapta el módulo de llamadas al LLM.
2. Prepara un JSON con 5-10 ejemplos "ideales".
3. Implementa lógica para seleccionar ejemplos según configuración.
4. Usa `langchain.llms.OpenAI` (u otro wrapper).
5. Crea un script/notebook que itere configuraciones y registre resultados.

**Entregables**

* Código comentado en español (CLI o notebook)
* JSON de ejemplos


#### 2. Proyecto: RAG Básico con LlamaIndex y LexiQuipu

**Objetivo**
Montar un pipeline de Retrieval‑Augmented Generation sobre un corpus breve usando LlamaIndex integrado con LexiQuipu.


**Descripción**

1. Indexa 3-5 documentos Markdown/TXT.
2. Genera embeddings (e.g. `text-embedding-ada-002`).
3. Flujo RAG: recupera top‑k (k=3), concatena al prompt y genera respuesta.
4. Análisis cualitativo de cómo varía la precisión al cambiar *k* y riesgos de bias.

**Pasos sugeridos**

1. Usa el módulo de embeddings de LexiQuipu o `openai.Embedding`.
2. Construye índice con LlamaIndex.
3. Ejecuta queries y registra respuestas.
4. Repite con k={1,3,5} y anota diferencias.

**Entregables**

* Script/notebook con indexación, queries y resultados para distintos *k*

#### 3. Proyecto: Tokenización y Embeddings Visuales

**Objetivo**
Explorar cómo diferentes tokenizadores afectan los embeddings y la similitud entre frases.

**Descripción**

1. Tokeniza un corpus de \~20 frases con Byte‑Pair Encoding y separación por espacios.
2. Genera embeddings (modelo Hugging Face).
3. Calcula similitud coseno entre frases y proyecta en 2D (TSNE o PCA).

**Pasos sugeridos**

1. Implementa tokenización con `tokenizers` o `sentencepiece`.
2. Usa Hugging Face para obtener embeddings.
3. Calcula matriz de similitud coseno.
4. Proyecta en 2D e interpreta clusters.

**Entregables**

* Script en Python con comentarios en español
* Gráfica 2D (TSNE/PCA)

#### 4. Proyecto: Prompting básico y control de salida

**Objetivo**
Analizar cómo ejemplos en el prompt moldean la respuesta de un LLM en tareas simples.


**Descripción**

1. Diseña tres versiones de prompt para una tarea (p. ej. resumen, clasificación de sentimientos).
2. Incluye variaciones con ejemplos "in‑context".
3. Compara respuestas en términos de precisión, coherencia y sesgos.

**Pasos sugeridos**

1. Prepara un notebook Jupyter.
2. Define la tarea y crea ejemplos.
3. Ejecuta los prompts contra el LLM y extrae métricas (longitud, ROUGE, exactitud).
4. Tabula y analiza diferencias.

**Entregables**

* Notebook con prompts, ejecuciones y métricas
* Tabla comparativa de resultados

#### 5. Proyecto: Mini RAG con LangChain y LlamaIndex

**Objetivo**
Implementar un flujo muy simple de Retrieval‑Augmented Generation usando LangChain/LlamaIndex.

**Descripción**

1. Indexa 2-3 documentos (PDF o markdown).
2. Recupera fragmentos top‑k y genera respuesta con el LLM.
3. Compara outputs con/sin recuperación.

**Pasos sugeridos**

1. Crea índice con LlamaIndex.
2. Monta chain en LangChain: retrieval + generation.
3. Ejecuta ejemplos y registra diferencias.
4. Resume en breve tabla comparativa.

**Entregables**

* Código (script/notebook) con indexación y queries
* Tabla comparativa de respuestas

#### 6. Proyecto: Agente simple para tareas web

**Objetivo**
Construir un agente de LangChain capaz de recuperar información de una API y ejecutar una acción.

**Descripción**

1. Implementa un agente que:

   * Pregunte al usuario por una ciudad.
   * Use API pública (p. ej. clima) para obtener datos.
   * Formatee la respuesta y la devuelva al usuario.
2. Demuestra interacción consola-agente.

**Pasos sugeridos**

1. Configura `LangChain` con un `Tool` para la API de clima.
2. Define la lógica de agente (`AgentExecutor`).
3. Prueba en consola con varios ejemplos de ciudad.
4. Documenta flujo de control.

**Entregables**

* Script Python comentado
* Ejemplo de interacción (logs de consola)

#### 7. Proyecto: Fine‑tuning  de un modelo pequeño

**Objetivo**
Fine‑tune rápidamente `distilbert-base-uncased` en una mini tarea de clasificación.


**Descripción**

1. Carga un dataset pequeño (p. ej. SST‑2 con 200 ejemplos).
2. Fine‑tune una época y evalúa en validación.
3. Registra métricas (accuracy) y pérdida vs. pasos.

**Pasos sugeridos**

1. Usa `transformers` + `Trainer` de Hugging Face.
2. Define `TrainingArguments` (learning rate, batch size).
3. Ejecuta entrenamiento de 1 época.
4. Grafica curva de pérdida y accuracy.

**Entregables**

* Notebook con código y comentarios en español
* Gráfica pérdida vs. pasos.

