## Miniproyectos CC0C2

>**Esta lista de proyectos corresponden a la práctica calificada 4 CC0C2**

**Estructura de evaluación sugerida:**
* **Proyecto (código, cuaderno de resultados):** Hasta 8 puntos.
* **Presentación (claridad, análisis, demostración, vídeo):** Hasta 12 punto (este punto es crucial y refleja la priorización de la exposición).
* **Entrega la dirección del repositorio de desarrollo**.
* **Fecha de presentación:** 3 de julio desde las 16:00 hrs.

#### Controles generales para todos los proyectos

1. **Pruebas de integridad de código**: scripts basados en AST (`tests/check_originality.py`) que evalúan similitudes estructurales con soluciones externas.
2. **Presentación y defensa**: demo en vivo o video corto donde el equipo explica el flujo interno.
3. **Callbacks y tests personalizados**: cada proyecto incluye ejercicios teóricos (`.md`) y pruebas unitarias que exigen respuestas y comportamientos personalizados.
4. **Fingerprinting de ejecución**: al guardar checkpoints o logs se añade un hash único (HMAC) vinculado al equipo.

### Proyecto 1: Adaptadores en PyTorch

**Contexto y motivación**
Los adaptadores (adapters) son pequeños módulos insertados en redes preentrenadas que permiten afinar modelos gigantes sin tener que volver a entrenar todos sus parámetros. Este proyecto invita al estudiante a comprender dos patrones básicos -bottleneck y paralelo- e integrarlos en un Transformer de propósito general, explorando el impacto en eficiencia y precisión.

**Objetivos**

1. Comprender la arquitectura básica de un Transformer y el concepto de "congelar" pesos.
2. Diseñar y aislar dos tipos de adaptadores:

   * **Bottleneck**: dos capas lineales con reducción e incremento de dimensión.
   * **Paralelo**: módulo que corre en paralelo a la capa original.
3. Insertar los adaptadores en cada bloque de atención y feed-forward usando hooks (`forward_hooks`).
4. Ejecutar entrenamientos donde solo los adaptadores sean entrenables; comparar con un entrenamiento "desde cero" y uno "full fine-tuning".
5. Medir y visualizar la relación entre número de parámetros entrenados, tiempo de convergencia y métrica de precisión (p. ej. exactitud en NER sintético).

**Entregables**

* `models/transformer_base.py`: implementación limpia de un Transformer secuencial.
* `models/adapter.py`: clases `AdapterBottleneck` y `AdapterParallel`, documentadas con docstrings.
* `hooks/forward_hooks.py`: funciones para registrar y activar/desactivar adaptadores dinámicamente.
* `train.py`: script principal con opciones `--adapter_type`, `--optimizer` y `--freeze_base`.
* `evaluate.py`: benchmarks automáticos sobre tareas de secuencia (clasificación simple) y NER sintético.
* `results/`: carpeta con:

  * gráficos de convergencia (pérdida y precisión) en formato PNG.
  * CSV con medidas de parámetros entrenados vs. exactitud.
* **Vídeo de demostración**: un clip de 6-10 min mostrando el pipeline de entrenamiento y evaluación (ejecución en terminal).

**Retos clave**

* Inicialización de pesos robusta (He vs. Xavier) y su impacto en la estabilidad.
* Uso de `dropout` y `weight_decay` solo sobre los adaptadores.
* Automatización de comparativas: escribir un pequeño script o Makefile que lance todos los experimentos y recolecte métricas.

### Proyecto 2: Implementación de LoRA en PyTorch

**Contexto y motivación**
LoRA (Low-Rank Adaptation) factoriza las actualizaciones de peso en matrices de rango reducido -ΔW = A·B permitiendo un fine-tuning eficiente en memoria y cómputo. El objetivo es replicar esta técnica desde cero y experimentar con sus parámetros fundamentales.

**Objetivos**

1. Crear un módulo `LoRALinear` que sustituya/rode las capas lineales estándar, inyectando matrices `A` (d×r) y `B` (r×k).
2. Desarrollar `LoRAAttention` extendiendo el mecanismo de atención multi-cabeza para incorporar LoRA en las proyecciones Q/K/V y la salida.
3. Garantizar que el `state_dict` permita separar y combinar los pesos base y los adaptadores LoRA, y soportar guardado/recarga con `merge_weights=True/False`.
4. Experimentar con distintos rangos `r` (por ejemplo, 4, 16, 64) y tasas de aprendizaje diferenciadas para A y B.
5. (Opcional) Integrar un simple esquema de pruning y cuantización usando operaciones nativas de PyTorch.

**Entregables**

* `models/lora.py`: definición de `LoRALinear` y `LoRAAttention`, con ejemplos de uso en el docstring.
* `utils/state_utils.py`: funciones `save_lora_state` y `load_lora_state` que manejan merges y splits de pesos.
* `scripts/train_lora.py`: pipeline de entrenamiento multiclasificación sobre un dataset pequeño (p. ej. IMDb).
* `scripts/pruning_quant.py`: script opcional para aplicar `torch.nn.utils.prune` y cuantización dinámica ligera.
* `analysis/plots.py`: generación de gráficas que relacionan parámetros entrenados vs. accuracy para cada `r`.
* **Vídeo de demostración**: grabación de pantalla de \~10 min mostrando

  - Inicialización del modelo con LoRA,
  - Ejecución de un lote de entrenamiento,
  - Visualización rápida de métricas en TensorBoard o consola.

**Retos clave**

* Coordinar formas de tensores al mezclar pesos base y LoRA.
* Ajustar learning rates separados y demostrar sus efectos en convergencia.
* Implementar pruning/quantización sin depender de librerías externas y comparar tiempos.


### Proyecto 3: Fine-tuning con Transformers y Hugging Face

**Contexto y motivación**
El ecosistema Hugging Face simplifica enormemente el fine-tuning de modelos pre-entrenados. Este proyecto guía al estudiante por todo el flujo: desde la tokenización hasta el entrenamiento supervisado con callbacks y métricas personalizadas.

**Objetivos**

1. Preprocesar un corpus pequeño (p. ej. reseñas de producto) usando `datasets` y el tokenizador "fast" elegido.
2. Configurar un `Trainer` con:

   * `TrainingArguments` parametrizables (batch size, learning rate, epochs).
   * Callbacks para early stopping y logging personalizado.
   * Métricas de precisión y recall usando `compute_metrics`.
3. Implementar AWP (Adversarial Weight Perturbation) y mixout de forma manual, sin librerías externas, dentro del loop de entrenamiento.
4. Añadir manejo de gradiente acumulado y detección de OOM con reinicio suave del entrenamiento.

**Entregables**

* `data/preprocess.py`: tokenización, limpieza y split del dataset.
* `training/trainer_config.py`: funciones que devuelven `TrainingArguments` y listas de callbacks.
* `training/train.py`: orquestador principal que carga data, modelo (`AutoModelForSequenceClassification`) y lanza `Trainer`.
* `evaluation/eval.py`: script para evaluar en test set, generar matriz de confusión y reportes de classification.
* `logs/`:

  * Checkpoints guardados automáticamente.
  * Archivos de TensorBoard (`.tfevents`).
* **Vídeo de demostración**: \~10 min mostrando la ejecución de `train.py`, visualización de métricas en TensorBoard y un ejemplo de inferencia con `pipeline`.

**Retos clave**

* Integrar AWP y mixout controlando que no interfieran con el scheduler de optimización.
* Diseñar un callback de early stopping sencillo.
* Balancear batch size y gradiente acumulado para evitar OOM en GPUs limitadas.


### Proyecto 4: Pipeline de inferencia y optimización

**Contexto y motivación**
Más allá del entrenamiento, desplegar modelos en producción exige optimizar latencia y throughput. Este proyecto cubre desde el uso de pipelines de Hugging Face hasta la exportación a ONNX y su integración en un microservicio.

**Objetivos**

1. Definir pipelines para:

   * **Generación de texto** (`pipeline("text-generation")`)
   * **Question Answering** (`pipeline("question-answering")`)
2. Exportar un modelo ligero (p. ej. DistilBERT) a ONNX (post-training) y comparar tiempos de inferencia en CPU.
3. Implementar dos esquemas de batching:

   * Token-a-token (autogreso incremental).
   * Lote completo (batch) con padding dinámico.
4. Desarrollar un servidor REST mínimo con FastAPI exponiendo `/generate` y `/qa`, midiendo latencia y throughput.

**Entregables**

* `inference/pipelines.py`: funciones que instancian y ejecutan ambos pipelines.
* `export/onnx_export.py`: código para conversión `model.to_onnx(...)` y recarga.
* `benchmarks/benchmark.py`: mediciones de tiempo por petición y por lote para CPU vs. ONNX.
* `server/app.py`: FastAPI con rutas documentadas y ejemplos en Swagger UI.
* **Vídeo de demostración**: \~10 min donde se muestra:

  - Exportación a ONNX,
  - Ejecución de `benchmark.py`,
  - Llamadas a la API con `curl` o Postman.

**Retos clave**

* Gestionar padding dinámico sin introducir overhead.
* Medir con precisión latencia vs. throughput y presentar resultados.
* Mantener el servidor lo más ligero posible, evitando frameworks pesados.

### Proyecto 5: Preentrenamiento ligero de un LLM

**Contexto y motivación**
Entrenar un gran modelo desde cero es inviable en recursos limitados, pero puede hacerse un preentrenamiento "ligero" sobre un corpus reducido para entender el pipeline completo de LM.

**Objetivos**

1. Limpiar un corpus de texto (p. ej. colección de artículos) y fragmentarlo en chunks de contexto fijo.
2. Implementar desde cero un `DataCollatorForLanguageModeling` que aplique masking dinámico (por token) y estático (por fragmento).
3. Definir un Transformer pequeño (`transformer_lm.py`) con 4-6 capas y dimensión reducida.
4. Entrenar en CPU o GPU modesta, registrando PPL (perplexity) y métricas de diversidad (`distinct-1`, `distinct-2`).
5. Evaluar generación de texto corto (párrafos de 50-100 tokens) y analizar coherencia.

**Entregables**

* `data/clean_corpus.py`: filtrado de caracteres, tokenización base y chunking en archivos JSONL.
* `data/collator.py`: clase `CustomDataCollator` con métodos de enmascarado.
* `models/transformer_lm.py`: modelo definido en PyTorch con docstrings.
* `train_pretrain.py`: loop con logging de pérdida, PPL y distinct-n cada epoch.
* `analysis/eval_generation.py`: scripts para generar muestras y calcular métricas de diversidad.
* **Vídeo de demostración**: \~10 min mostrando la preparación de datos, inicio de entrenamiento y ejemplos de generación.

**Retos clave**

* Afinar ratio de mask (p. ej. 15 %) y tamaño de ventana de contexto.
* Minimizar consumo de memoria en DataLoader.
* Evaluar coherencia frente a diversidad de las muestras.


### Proyecto 6: Alineación con RLHF básico

**Contexto y motivación**
El entrenamiento con feedback humano (RLHF) consta de tres fases: creación de un reward model, generación de preferencias y optimización de la política por PPO. Este proyecto cubre un flujo mínimo en PyTorch.

**Objetivos**

1. Generar un dataset sintético de pares (respuesta A vs. B) con indicación de preferencia.
2. Entrenar un reward model sencillo (clasificador binario sobre embeddings).
3. Implementar desde cero un bucle PPO (clipped), controlando la penalización KL.
4. Monitorizar estabilidad de la política (reward promedio, divergencia KL) durante el entrenamiento.

**Entregables**

* `data/preferences.py`: código para crear y guardar un CSV/JSONL con ejemplos sintéticos.
* `models/reward_model.py`: clase `RewardModel` basada en un Transformer congelado + capa linear.
* `rl/ppo.py`: implementación de PPO (computación de ventajas, ratio, clipping).
* `train_rlhf.py`: orquestación del loop completo.
* `analysis/metrics_rl.py`: scripts que plotean reward y KL por iteración.
* **Vídeo de demostración**: \~10 min enseñando generación de ejemplos, entranamiento del reward model y loop PPO.

**Retos clave**

* Calcular ventajas (GAE) y clipping de ratio correctamente.
* Balancear learning rate del policy vs. reward model.
* Controlar sesgos introducidos por datos sintéticos de preferencia.


### Proyecto 7: Agentes RAG y LangChain simplificado

**Contexto y motivación**
La generación aumentada por recuperación (RAG) combina indexación de vectores y generación de texto. Aquí construiremos un pipeline manual con FAISS y, opcionalmente, un flujo muy básico de LangChain.

**Objetivos**

1. Indexar un pequeño conjunto de documentos (p. ej. párrafos) usando FAISS puro en CPU.
2. Definir un módulo `RagRetriever` que reciba una consulta, recupere vectores y retorne fragmentos relevantes.
3. Implementar un generador simple que concatene contexto + pregunta y llame a un `pipeline("text-generation")`.
4. (Opcional) Montar un flujo de LangChain muy acotado usando `LLMChain` y un buffer de conversación ligero.
5. Probar herramientas "mock" integradas en el agente.

**Entregables**

* `retriever/faiss_index.py`: construcción del índice y funciones de búsqueda.
* `models/rag_module.py`: clase que une retriever + generador.
* `agents/langchain_flow.py`: script que define un objeto `LLMChain` minimalista y estado de memoria.
* `tools/tool_stub.py`: ejemplo de herramienta externa (p. ej. búsqueda web simulada).
* `run_agent.py`: CLI que permite interacciones iterativas con el agente.
* **Vídeo de demostración**: \~10 min mostrando indexación, ejecución de consultas y llamadas a herramientas mock.

**Retos clave**

* Manejar dependencias de FAISS en CPU sin errores de instalación.
* Diseñar prompts modulares fácilmente extensibles.
* Mantener un buffer de conversación ligero y persistencia mínima (p. ej. en JSON).



