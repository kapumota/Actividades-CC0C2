### **Práctica calificada 5 CC0C2**

#### Examen 1: Preentrenamiento ligero de un LLM  

**Duración:** 3 horas  
**Repositorio base:** [nlp-proyecto05](https://github.com/GabrielaC01/nlp-proyecto05)

#### Preparación de datos  
- Escribe un script que lea un corpus de artículos en texto plano, aplique filtrado de caracteres no ASCII y genere chunks de longitud fija (p. ej. 512 tokens) en formato JSONL.  
- Menciona cómo ajustarías el ratio de chunking para balancear cobertura vs. memoria.

#### Implementación de CustomDataCollator  
En `data/collator.py`, implementa dos métodos:  
1. Masking dinámico (por token, ratio 15 %).  
2. Masking estático (por fragmento: enmascara un 10 % de los chunks completos).  
- Explica brevemente el trade-off memoria vs. variedad de contextos.

#### Definición del modelo  
En `models/transformer_lm.py`:  
- Crea un Transformer con 4 capas y `d_model=256`.  
- Añade docstrings que expliquen cada componente (embeddings, capas de atención, feed-forward).

#### Loop de entrenamiento y métricas  
Modifica `train_pretrain.py` para:  
- Loguear pérdida y PPL tras cada época.  
- Calcular métricas de diversidad `distinct-1` y `distinct-2` al final de cada epoch.  
- Inserta un callback para guardar el mejor checkpoint según PPL.

#### Análisis de generación  
- Usa `analysis/eval_generation.py` para generar 5 párrafos de 50–100 tokens.  
- Calcula y compara PPL vs. `distinct-2`; comenta brevemente la coherencia de las muestras.


#### Examen 2: Fine-tuning de Transformers con Hugging Face y Técnicas Avanzadas  
**Duración:** 3 horas  
**Repositorio base:** [nlp-proyecto3-pc4-pc5](https://github.com/Cleber96/nlp-proyecto3-pc4-pc5.git)

##### Tokenización y DataLoader  
- En `data/preprocess.py`, ajusta la tokenización para truncar a 128 tokens y pad dinámico.  
- Describe cómo habilitarías `gradient_accumulation` en un entorno con 4 GB de GPU.

#### Configurar Trainer con adapters y LoRA  
En `training/trainer_config.py`:  
- Añade una configuración de LoRA (`rank=8`, `α=16`) aplicada a las capas de atención.  
- Un Adapter (`bottleneck=64`) en cada capa feed-forward.  
- Define cómo en el loop de entrenamiento se habilitan solo estos parámetros para optimización.

#### Implementación de AWP y mixout manual  
En `training/train.py`, dentro de `compute_loss` o hook de backward:  
- Aplica AWP con `ε=0.01` durante la mitad de cada epoch.  
- Integra mixout con `p=0.1` en las capas lineales del classifier.

#### Callbacks y métricas  
- Crea un callback sencillo para early stopping tras 3 epochs sin mejora en F1.  
- Ajusta `compute_metrics` para devolver precision, recall y F1, y muestra la matriz de confusión tras el entrenamiento.

#### Evaluación final  
- Ejecuta `evaluation/eval.py` sobre el test set y captura el reporte de clasificación.  
- Redacta en 5 líneas cómo afectarían los adapters y LoRA a la latencia de inferencia.

#### Examen 3: Pipeline de inferencia y optimización  
**Duración:** 3 horas  
**Repositorio base:** [nlp-proyecto4](https://github.com/TutMosis22/nlp-proyecto4.git)

#### Definición de pipelines  
En `inference/pipelines.py`, implementa dos funciones:  
1. `generate_text(prompt: str) -> str` usando `pipeline("text-generation")` con `top-k=50`.  
2. `answer_question(context: str, question: str) -> str` con `pipeline("question-answering")`.

#### ONNX Export & Benchmark  
En `export/onnx_export.py`:  
- Escribe el código para convertir un modelo DistilBERT a ONNX con export dinámico de ejes.  
- Cargarlo de nuevo y generar una predicción simple.

En `benchmarks/benchmark.py`:  
- Mide y compara latencia (ms) de CPU vs. ONNX (100 requests).

#### Batching token-a-token vs. dinámico  
- Implementa en `inference/pipelines.py` dos esquemas de batching:  
  - Token-a-token: envía cada token de la generación incrementalmente.  
  - Batch completo: agrupa N prompts con padding dinámico.  
- Mide el throughput (tokens/s) de ambos con 20 prompts.

#### Servidor FastAPI  
En `server/app.py`:  
- Crea rutas `/generate` y `/qa`.  
- Documenta los esquemas de batching y ONNX en Swagger UI.  
- Asegura que el servidor arranca en < 1 s y responde en < 200 ms para un prompt de 50 tokens.

#### Informe breve  
Redacta un `README.md` corto que explique:  
- Cómo replicar benchmarks.  
- Las ventajas de ONNX y padding dinámico.  
- Posibles mejoras (p. ej. quantization int8).

#### Examen 4: Agentes RAG y LangChain simplificado  
**Duración:** 3 horas  
**Repositorio base:** [nlp-proyecto-07](https://github.com/franklinep/nlp-proyecto-07)

#### Construcción de índice FAISS  
En `retriever/faiss_index.py`:  
- Indexa 1 000 párrafos en vectores de 768 dim.  
- Añade un parámetro para re-indexar con Opportunistic Product Quantization (OPQ) y compara tamaño en MB.

#### Módulo RagRetriever y adapters  
En `models/rag_module.py`:  
- Define `RagRetriever` que reciba consulta y devuelva top k fragmentos.  
- Integra un Adapter ligero (`bottleneck=32`) en el proyector de consulta antes de vectorizar.

#### Generador y LoRA fine-tuning  
- Crea `run_lora_finetune.py` que cargue un LLM base (gpt2) y aplique LoRA (`rank=4`) en las capas de atención.  
- Fine-tune con pares sinteticos (500 ejemplos).  
- Evalúa la puntuación de ROUGE-L antes y después del fine-tuning.

#### Flujo LangChain mínimo  
En `agents/langchain_flow.py`:  
- Monta un `LLMChain` con prompt template modular que inserte los fragmentos RAG.  
- Un buffer de conversación de longitud máxima 3.  
- Simula una herramienta externa `web_search` y muestra cómo se integra.

#### CLI interactivo y demo  
En `run_agent.py`:  
- Implementa un CLI REPL que pregunte al usuario por input y muestre respuestas RAG+LLM.  
- Permite invocar la `web_search` mock con `!search término`.  
- Graba un snippet de terminal (captura de pantalla incluida) con al menos 3 interacciones.

#### Entrega

Cada estudiante presentará su propio repositorio con todos los scripts modificados, los resultados (tablas, gráficas, checkpoints) y un informe en Markdown que documente brevemente la instalación, la ejecución y un análisis de los resultados obtenidos.

#### Puntuaciones

#### Examen 1: Preentrenamiento ligero de un LLM (20 pt)  
- **Preparación de datos y chunking (4 pt)**
  - 2 pt: Script `data/clean_corpus.py` que genera JSONL correctamente.  
  - 2 pt: Calidad del chunking (ejemplos claros, justificación de ratios).
- **Implementación de CustomDataCollator (4 pt)**
  - 2 pt: Masking dinámico correctamente aplicado.  
  - 2 pt: Masking estático implementado y documentado.
- **Definición del modelo (3 pt)**
  - 2 pt: Arquitectura Transformer de 4 capas y `d_model=256`.  
  - 1 pt: Docstrings claros y consistentes.
- **Loop de entrenamiento y métricas (5 pt)**
  - 2 pt: Logging de pérdida y PPL cada epoch.  
  - 2 pt: Cálculo de `distinct-1` y `distinct-2`.  
  - 1 pt: Guardado de checkpoint según PPL.
- **Análisis de generación (4 pt)**
  - 2 pt: Generación de 5 párrafos de 50–100 tokens.  
  - 2 pt: Informe en Markdown comentando coherencia vs. diversidad (métricas).

#### Examen 2: Fine-tuning con Transformers y técnicas avanzadas (20 pt)  
- **Preprocesamiento y DataLoader (3 pt)**
  - 2 pt: Tokenización a 128 tokens y padding dinámico.  
  - 1 pt: Uso adecuado de `gradient_accumulation` en entorno limitado.
- **Configuración de LoRA y Adapters (5 pt)**
  - 3 pt: Parámetros de LoRA (rank, α) aplicados correctamente.  
  - 2 pt: Adapters integrados en capas feed-forward.
- **Implementación de AWP y mixout (4 pt)**
  - 2 pt: AWP en el loop de entrenamiento.  
  - 2 pt: Mixout incorporado en capas lineales.
- **Callbacks y métricas (4 pt)**
  - 2 pt: Callback de early stopping funcionando.  
  - 2 pt: Métricas de precision, recall y F1, y matriz de confusión.
- **Análisis de resultados (4 pt)**
  - 2 pt: Tabla comparativa de parámetros entrenables vs. full fine-tuning.  
  - 2 pt: Comentario sobre impacto en latencia y uso de GPU.

#### Examen 3: Pipeline de Inferencia y Optimización (20 pt)  
- **Definición de pipelines (3 pt)**
  - 2 pt: Funciones `generate_text` y `answer_question` correctas.  
  - 1 pt: Ejemplos de uso reproducibles.
- **Export a ONNX y benchmark (5 pt)**
  - 3 pt: Script de export (`export/onnx_export.py`) validado.  
  - 2 pt: Benchmark de latencia CPU vs. ONNX (resultados claros).
- **Implementación de batching (4 pt)**
  - 2 pt: Token-a-token vs. batch dinámico implementados.  
  - 2 pt: Medición de throughput y comparación.
- **Servidor FastAPI (4 pt)**
  - 2 pt: Endpoints `/generate` y `/qa` funcionando.  
  - 2 pt: Documentación en Swagger UI y evidencias de tiempos.
- **README de replicación (4 pt)**
  - 2 pt: Instrucciones claras para reproducir benchmarks.  
  - 2 pt: Sugerencias de mejoras (quantization, optimizaciones).

#### Examen 4: Agentes RAG y LangChain simplificado (20 pt)  
- **Índice FAISS y OPQ (4 pt)**
  - 2 pt: Construcción de índice FAISS en CPU.  
  - 2 pt: Comparativa de tamaño con OPQ.
- **Módulo RagRetriever con Adapter (4 pt)**
  - 2 pt: Recuperación top-k correcta.  
  - 2 pt: Adapter integrado y justificado.
- **Fine-tuning con LoRA (4 pt)**
  - 2 pt: Script `run_lora_finetune.py` válido.  
  - 2 pt: Logs de entrenamiento y evaluación ROUGE-L antes/después.
- **Flujo LangChain mínimo (4 pt)**
  - 2 pt: `LLMChain` con buffer de conversación.  
  - 2 pt: Integración de herramienta mock `web_search`.
- **CLI interactivo y demostración (4 pt)**
  - 2 pt: `run_agent.py` con REPL funcional.  
  - 2 pt: Captura de terminal con al menos 3 interacciones.
