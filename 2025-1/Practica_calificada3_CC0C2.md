### **Práctica calificada 3 CC0C2**

- **Fecha de Entrega:** 19 de junio
- **Modalidad:** Grupal (máximo 2 integrantes)
- **Referencia general:** ["The Transformer Family Version 2.0"](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) de Lilian Weng

#### **Instrucciones generales**

Cada grupo deberá seleccionar **uno** de los siguientes proyectos. El objetivo es profundizar en arquitecturas de Transformers más allá del modelo base, explorando optimizaciones y aplicaciones novedosas. Todos los proyectos deben cumplir con los siguientes requisitos:

* **Implementación:** El código debe ser desarrollado en **PyTorch**, superando las **500 líneas de código** (excluyendo comentarios y boilerplate). El código debe estar bien comentado y estructurado.
* **Repositorio:** El proyecto completo debe ser entregado a través de un repositorio de **Git**  que incluya un archivo `README.md` detallado con:
    * Descripción del proyecto y objetivos.
    * Instrucciones para instalar dependencias (`requirements.txt`).
    * Guía para ejecutar el código y reproducir los resultados.
    * Verificación de trabajo en conjunto y entrega individual de repositorios
* **Presentación:** Se realizará una presentación de **20 minutos** por grupo, dividida en:
    * **Contexto teórico:** ¿Qué problema se aborda y cuál es la idea fundamental?
    * **Detalles de implementación:** Arquitectura, decisiones clave y desafíos.
    * **Resultados y análisis:** Gráficos, tablas y conclusiones.
    * **Lecciones aprendidas:** ¿Qué funcionó, qué no y qué se haría diferente?

> Es requisito la entrega del proyecto para poder realizar la presentación. La evaluación es de tipo expositiva.

**Evaluación**

- Trabajo presentado: 8 puntos
- Exposición: 12 puntos

### **Lista de proyectos a elegir**

A continuación se detallan los 8 proyectos. Escoge solo uno.

#### **1. Máscara dinámica y entrenamiento autoregresivo**

* **Contexto teórico:**
    Los modelos de lenguaje generativos, como GPT, funcionan de manera **autorregresiva**: generan texto una palabra (o token) a la vez, basándose en las que ya han generado. Para entrenar un Transformer de manera eficiente en esta tarea, se utiliza una **máscara causal (o de atención triangular)**. Esta máscara asegura que, al predecir el token en la posición `i`, el modelo solo pueda atender a los tokens anteriores (de `0` a `i-1`), evitando que "vea el futuro". En la fase de inferencia, se emplean técnicas de muestreo como **temperatura** y **top-k** para controlar la aleatoriedad y la calidad del texto generado. Una temperatura alta produce texto más sorprendente pero propenso a errores, mientras que un top-k bajo limita las opciones a las más probables, generando texto más coherente pero menos diverso.

* **Objetivos específicos:**
    -  Implementar un decoder de Transformer desde cero, prestando especial atención a la creación y aplicación de la máscara causal en el mecanismo de self-attention.
    -  Desarrollar funciones para la generación de secuencias token a token.
    -  Integrar el **muestreo por temperatura** para ajustar la distribución de probabilidad de los tokens de salida.
    -  Implementar el **muestreo top-k** para restringir el vocabulario de muestreo a los `k` tokens más probables.

* **Entregables clave:**
    * **Código y notebook:** Un notebook que muestre claramente:
        * La implementación del decoder con su máscara causal.
        * Ejemplos de texto generado usando diferentes configuraciones de temperatura (e.g., T=0.2, T=0.8, T=-2) y top-k (e.g., k=5, k=50).
    * **Gráficos:** Un gráfico de dispersión que relacione la **temperatura** (eje X) con una métrica de **diversidad del texto** (e.g., número de n-gramas únicos, eje Y) para mostrar empíricamente cómo la creatividad aumenta con la temperatura.

#### **2. Transformer con atención dispersa (Sparse Attention)**

* **Contexto teórico:**
    El principal cuello de botella computacional de un Transformer es la matriz de atención, cuya complejidad es $O(L^2)$, donde $L$ es la longitud de la secuencia. Esto hace que procesar secuencias muy largas (libros, genomas, audio de alta resolución) sea inviable. La **atención dispersa (Sparse Attention)** propone reemplazar esta matriz densa por una matriz dispersa. En lugar de que cada token atienda a todos los demás, solo atiende a un subconjunto. Un patrón común es la **atención de ventana deslizante (block sparse)**, donde cada token solo atiende a los `w` tokens vecinos, reduciendo la complejidad a $O(L \cdot w)$.

* **Objetivos específicos:**
    -  Modificar la capa de atención estándar de un Transformer para implementar un patrón de atención dispersa de ventana local.
    -  Diseñar un mecanismo eficiente para aplicar la máscara de dispersión sin construir explícitamente la matriz $L \times L$.
    -  Evaluar el impacto de esta optimización en la velocidad y el rendimiento del modelo.

* **Entregables clave:**
    * **Benchmarks de tiempo:** Código que mida y compare el tiempo de ejecución (forward pass) de una capa de atención estándar vs. tu implementación de atención dispersa para secuencias de longitud creciente (e.g., L=512, 1024, 2048, 4096).
    * **Tabla comparativa:** Una tabla que resuma los resultados en un dataset de lenguaje estándar (e.g., WikiText-2), comparando el modelo base con tu modelo optimizado en:
        * **Perplejidad (PPL):** Métrica de calidad del lenguaje.
        * **Throughput:** Secuencias procesadas por segundo.

#### **3. Adaptive attention span**

* **Contexto teórico:**
    No todas las cabeceras de atención (attention heads) en un Transformer necesitan la misma cantidad de contexto. Algunas pueden especializarse en patrones locales (sintaxis), mientras que otras pueden necesitar un contexto más amplio (semántica). El **Adaptive Attention Span** es una técnica que permite al modelo **aprender** el tamaño de contexto óptimo (el "span") para cada cabecera de atención. Esto se logra parametrizando el tamaño de la ventana de atención con un valor $\alpha_h$ para cada cabecera $h$, el cual es entrenable vía gradiente. El resultado es un modelo más eficiente y, a menudo, más interpretable.

* **Objetivos específicos:**
    -  Implementar una capa de atención donde el tamaño de la ventana de atención no es fijo, sino un parámetro aprendible.
    -  Añadir un término de regularización al loss para incentivar spans más cortos y promover la eficiencia.
    -  Visualizar cómo diferentes cabeceras aprenden diferentes spans a lo largo del entrenamiento.

* **Entregables clave:**
    * **Gráficos de spans:** Un gráfico de barras o un mapa de calor que muestre el valor final del "span" aprendido por cada cabecera de atención en cada capa del Transformer.
    * **Análisis calidad vs. span:** Un análisis que discuta la relación entre el span promedio aprendido y la calidad final del modelo (PPL). ¿Se puede lograr un rendimiento similar con un contexto significativamente reducido?


#### **4. Depth-adaptive Transformer (DA-Transformer)**

* **Contexto teórico:**
    En un Transformer estándar, cada token de entrada debe pasar por el mismo número de capas (la "profundidad" del modelo), sin importar si es una palabra simple o una compleja. El **Depth-Adaptive Transformer (DA-Transformer)** introduce un mecanismo de **detención temprana (early exiting)**. En cada capa, un pequeño clasificador (halting ponderer) decide si el token ya ha sido procesado lo suficiente o si necesita pasar a la siguiente capa. Esto permite que el modelo ajuste dinámicamente la cantidad de cómputo por token, ahorrando recursos en las entradas más "fáciles".

* **Objetivos específicos:**
    -  Aumentar el Transformer con un mecanismo de "ponderación de detención" (halting ponderer) en cada capa.
    -  Implementar la lógica que permite a cada token "salir" del procesamiento en diferentes profundidades.
    -  Medir la eficiencia computacional ganada con esta técnica.

* **Entregables clave:**
    * **Histograma de capas:** Un histograma que muestre la distribución de la profundidad de "salida" de los tokens en un conjunto de datos de evaluación. ¿La mayoría de los tokens salen temprano o tarde?
    * **Discusión de resultados:** Un análisis cualitativo que investigue si hay una correlación entre el tipo de token (e.g., stop-words vs. sustantivos importantes) y el número de capas que utiliza.


#### **5. Low-Rank attention via factorización**

* **Contexto teórico:**
    La matriz de atención $A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ es computacionalmente cara. La hipótesis de la **atención de bajo rango (Low-Rank Attention)** es que esta matriz no necesita tener rango completo para ser efectiva; su información esencial puede ser capturada por una aproximación de rango mucho menor. Una forma de lograrlo es mediante la **factorización de matrices**, aproximando las matrices de peso $W_Q, W_K, W_V$ o la propia matriz de atención con productos de matrices más pequeñas. Esto reduce drásticamente el número de parámetros y el costo de la multiplicación de matrices.

* **Objetivos específicos:**
    -  Implementar una capa de atención donde las matrices de proyección (Query, Key, Value) sean factorizadas en dos matrices de menor rango ($W \approx W_1 W_2$).
    -  Diseñar la capa para que el rango `r` sea un hiperparámetro configurable.
    -  Comparar el rendimiento de esta aproximación frente a la atención estándar.

* **Entregables clave:**
    * **Benchmarks:** Tablas que comparen el modelo base con tu modelo de bajo rango (para diferentes valores de `r`) en:
        * **Tamaño del modelo:** Número total de parámetros.
        * **Calidad:** Perplejidad o accuracy en una tarea de clasificación.
    * **Código modular:** La implementación de la atención de bajo rango debe ser una clase modular que pueda reemplazar fácilmente a `torch.nn.MultiheadAttention`.


#### **6. Integración de transformer en RL (Decision Transformer)**

* **Contexto teórico:**
    El Aprendizaje por Refuerzo (RL) tradicionalmente usa redes como MLP o CNN para aprender políticas. El **Decision Transformer** reformula el RL como un problema de **modelado de secuencias**. En lugar de aprender una política que mapea *estado* a *acción*, modela una trayectoria completa como una secuencia de `(retorno_esperado_1, estado_1, acción_1, retorno_esperado_2, estado_2, acción_2, ...)`. Dado un retorno objetivo y el estado actual, el modelo predice la siguiente acción autoregresivamente. Esto permite usar el poder de los Transformers para resolver tareas de RL sin necesidad de algoritmos complejos como Q-Learning o Policy Gradients.

* **Objetivos específicos:**
    -  Adaptar la arquitectura de un Transformer para procesar secuencias de tuplas (retorno, estado, acción).
    -  Implementar el bucle de entrenamiento usando un dataset offline de trayectorias preexistentes.
    -  Evaluar la política aprendida en un entorno de simulación.

* **Entregables clave:**
    * **Código de entrenamiento y evaluación:** Un script completo que entrene el Decision Transformer en un entorno como **CartPole** (de OpenAI Gym) o uno de **Atari** (usando datos de D4RL).
    * **Gráficas de retorno:** Una gráfica que muestre el **retorno promedio obtenido** (eje Y) por el agente en el entorno a medida que se le proporcionan diferentes **retornos objetivo** en la inferencia (eje X).

#### **7. Transformer multimodal texto-imágenes (Vision-Language)**

* **Contexto teórico:**
    Los Transformers pueden procesar más que solo texto. Los modelos **multimodales** combinan información de diferentes fuentes, como texto e imágenes. Un enfoque común es:
    -  Usar un **Vision Transformer (ViT)** para dividir una imagen en `patches` (parches) y convertirlos en una secuencia de embeddings.
    -  Usar un Transformer de texto para convertir una oración en otra secuencia de embeddings.
    -  Fusionar estas dos secuencias usando un mecanismo de **cross-attention**, donde una modalidad (e.g., texto) genera las Queries y la otra (e.g., imagen) genera las Keys y Values. Esto permite al modelo "preguntar" sobre la imagen usando el contexto del texto.

* **Objetivos específicos:**
    -  Implementar un pipeline que procese tanto imágenes como texto.
    -  Utilizar embeddings pre-entrenados de un ViT (o implementar uno simple) y un tokenizer de texto.
    -  Construir un decoder que combine ambos tipos de embeddings a través de capas de cross-attention para generar descripciones de imágenes (image captioning).

* **Entregables clave:**
    * **Ejemplos de captions:** En un notebook, mostrar varias imágenes de un conjunto de prueba (e.g., COCO) junto con las descripciones (captions) generadas por tu modelo.
    * **Diagramas del pipeline:** Un diagrama claro (puede ser hecho con herramientas como `draw.io`) que ilustre el flujo de datos: desde la imagen y el texto de entrada, pasando por los encoders y el decoder con cross-attention, hasta el texto de salida.

#### **8. Fine-tuning y pruning en Transformer**

* **Contexto teórico:**
    Los modelos de Transformer grandes (como BERT) son muy potentes pero también muy pesados para desplegar en dispositivos con recursos limitados. El **pruning (poda)** y la **cuantización** son técnicas para comprimir estos modelos.
    * **Pruning estructural:** En lugar de eliminar pesos individuales (poda no estructurada), se eliminan componentes enteros del modelo, como cabeceras de atención completas o neuronas en las capas FFN. Esto preserva la estructura densa de las matrices y acelera la inferencia en hardware estándar.
    * **Cuantización:** Consiste en reducir la precisión numérica de los pesos del modelo, por ejemplo, pasando de `float32` (32 bits) a `int8` (8 bits). Esto reduce el tamaño del modelo 4 veces y puede acelerar los cálculos.

* **Objetivos específicos:**
    -  Tomar un modelo Transformer pre-entrenado y pequeño, como **TinyBERT** o DistilBERT.
    -  Realizar un fine-tuning en una tarea de clasificación (e.g., GLUE).
    -  Aplicar un algoritmo de **pruning estructural** para eliminar un porcentaje de las cabeceras de atención menos importantes.
    -  Aplicar **cuantización post-entrenamiento** al modelo podado.

* **Entregables clave:**
    * **Comparativa de tamaños y accuracy:** Una tabla que compare cuatro versiones del modelo:
        -  Modelo original.
        -  Modelo con fine-tuning.
        -  Modelo podado.
        -  Modelo podado y cuantizado.
        La tabla debe mostrar el **tamaño en disco (MB)** y la **precisión** en el conjunto de prueba para cada uno.
    * **Script de exportación:** Un script que guarde el modelo final (cuantizado) en un formato optimizado para inferencia, como **ONNX** o **TorchScript**.
 

#### **9. Mixture of experts (MoE) para escalabilidad del Transformer**

* **Contexto teórico:**
    Una de las principales barreras para crear modelos de lenguaje cada vez más grandes es el costo computacional: en un Transformer estándar, todos los parámetros son utilizados para procesar cada token. La arquitectura **Mixture of Experts (MoE)** rompe con este paradigma. En lugar de una única y densa capa Feed-Forward (FFN), se utilizan múltiples FFN "expertas" en paralelo. Para cada token, una pequeña red "gating" o de enrutamiento elige dinámicamente qué experto (o un pequeño subconjunto de expertos) debe procesarlo. De esta manera, es posible escalar el número de parámetros del modelo a billones sin aumentar el costo computacional por token, ya que solo una fracción de los parámetros se activa en cada paso. Modelos como Mixtral 8x7B han demostrado la eficacia de este enfoque.

* **Objetivos específicos:**
    -  Implementar una capa MoE para reemplazar la subcapa FFN de un bloque de Transformer.
    -  Desarrollar la red "gating" que aprende a enrutar los tokens hacia los expertos más adecuados.
    -  Incorporar una **pérdida auxiliar de balanceo (auxiliary load balancing loss)**, crucial para evitar que la red sobre-especialice a unos pocos expertos mientras ignora a los demás.
    -  Analizar el comportamiento del enrutamiento y el impacto en el rendimiento.

* **Entregables clave:**
    * **Código de la capa MoE:** Una implementación modular de la capa `MoE` con su `gating network`.
    * **Gráfico de utilización de expertos:** Un mapa de calor o gráfico de barras que muestre la distribución de tokens asignados a cada experto a lo largo del entrenamiento, demostrando que la pérdida de balanceo está funcionando.
    * **Análisis comparativo:** Una discusión sobre la calidad (PPL) vs. la eficiencia computacional (FLOPs por token) del modelo MoE en comparación con un modelo denso de tamaño computacional similar.

#### **10. Generación aumentada por recuperación (RAG) con Transformers**

* **Contexto teórico:**
    El conocimiento de un modelo de lenguaje está limitado a la información presente en sus datos de entrenamiento (conocimiento paramétrico). Esto lo hace propenso a "alucinar" hechos y a quedar desactualizado. La **Generación Aumentada por Recuperación (RAG)** es una técnica que mitiga este problema al permitir que el modelo consulte una base de conocimiento externa y actualizada en tiempo de inferencia. El proceso típico implica:
    -  **Recuperador (Retriever):** Ante una pregunta, este componente busca y extrae los fragmentos de texto más relevantes de una base de datos vectorial (e.g., un índice de Wikipedia).
    -  **Generador (Generator):** El texto recuperado se concatena con la pregunta original y se le entrega como contexto a un modelo de lenguaje generativo (como GPT o T5) para que formule una respuesta informada y fundamentada.

* **Objetivos específicos:**
    -  Construir un pipeline de RAG completo.
    -  Utilizar un modelo de embeddings pre-entrenado (e.g., `sentence-transformers`) para indexar un corpus de documentos (puedes usar un subconjunto de Wikipedia o artículos de un tema específico) en una base de datos vectorial como FAISS o ChromaDB.
    -  Implementar la lógica que, dada una consulta, recupera los `k` documentos más relevantes.
    -  Integrar el contexto recuperado con un LLM pre-entrenado (e.g., `Flan-T5`, `GPT-2`) para generar la respuesta final.

* **Entregables clave:**
    * **Notebook demostrativo:** Un notebook que muestre el pipeline completo en acción. Debe incluir ejemplos claros donde se comparen las respuestas de un LLM base vs. las respuestas del sistema RAG a preguntas fácticas.
    * **Diagrama de arquitectura:** Un diagrama claro del sistema, ilustrando el flujo desde la consulta del usuario, pasando por la base de datos vectorial, hasta la generación final de la respuesta.
    * **Análisis cualitativo:** Una discusión sobre los tipos de preguntas donde RAG ofrece una mejora más significativa y los posibles puntos de fallo del sistema (e.g., recuperación de documentos irrelevantes).

#### **11. Modelos de espacio de estados (Mamba/S4) como alternativa a Transformers**

* **Contexto teórico:**
    A pesar de su éxito, la complejidad cuadrática de la atención en los Transformers ($O(L^2)$) sigue siendo un problema para secuencias muy largas. Recientemente, han surgido los **Modelos de Espacio de Estados (SSMs)** como una alternativa prometedora. Arquitecturas como **Mamba** y **S4** se inspiran en la teoría de control clásica y modelan secuencias a través de un estado latente recurrente. Su principal ventaja es que pueden ser formulados para operar de dos maneras: una versión recurrente para una inferencia autoregresiva ultra-rápida ($O(L)$) y una versión convolucional para un entrenamiento paralelo y eficiente (también $O(L \log L)$ o $O(L)$). Mamba mejora esto con un mecanismo de selección que permite al modelo enfocarse o ignorar selectivamente la información según el token de entrada, dándole capacidades similares a la atención pero con una complejidad lineal.

* **Objetivos específicos:**
    -  Investigar la formulación matemática de los SSMs, específicamente la transición entre la vista recurrente y la convolucional.
    -  Implementar un bloque **S4 (Structured State Space)** o un **bloque Mamba simplificado** desde cero. Esto implica implementar la discretización del estado continuo y el cálculo eficiente de la convolución.
    -  Reemplazar los bloques de atención y FFN de un Transformer por tu bloque SSM.
    -  Comparar la eficiencia y el rendimiento de tu modelo SSM con un Transformer estándar en una tarea de modelado de secuencias largas.

* **Entregables clave:**
    * **Código del bloque SSM:** La implementación del núcleo del modelo SSM, bien comentada para explicar la matemática subyacente.
    * **Benchmark de velocidad y memoria:** Gráficos que comparen el tiempo de inferencia y el uso de memoria de tu SSM vs. un Transformer estándar a medida que aumenta la longitud de la secuencia (e.g., 1k, 4k, 16k tokens), demostrando la ventaja de la complejidad lineal.
    * **Tabla comparativa de rendimiento:** Una tabla que compare la perplejidad (PPL) en un dataset de lenguaje (como WikiText-103) para evaluar si la eficiencia se logra sin un sacrificio significativo en la calidad.

### **Referencias**

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Holtzman, A., et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR.
3. Fan, A., et al. (2018). *Hierarchical Neural Story Generation*. ACL.
4. Child, R., et al. (2019). *Generating Long Sequences with Sparse Transformers*. arXiv:190-10509.
5. Zaheer, M., et al. (2020). *Big Bird: Transformers for Longer Sequences*. NeurIPS.
6. Sukhbaatar, S., et al. (2019). *Adaptive Attention Span in Transformers*. arXiv:190-07799.
7. Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural Networks*. arXiv:160-0898-
8. Katharopoulos, A., et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.
9. Zhao, B., et al. (2021). *Efficient Low-Rank Attention via Scalable Kernel Decomposition*. arXiv:210-1389-
10. Chen, L., et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling*. NeurIPS.
11. Dosovitskiy, A., et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR.
12. Lu, J., et al. (2022). *Unified Vision-Language Pre-Training for Image Captioning and VQA*. AAAI.
13. Michel, P., et al. (2019). *Are Sixteen Heads Really Better than One?* NeurIPS.
1- Jacob, B., et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR.
