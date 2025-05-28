## Miniproyectos CC0C2

**Estructura de evaluación sugerida:**
* **Proyecto (código, cuaderno de resultados):** Hasta 2 puntos.
* **Presentación (claridad, análisis, demostración):** Hasta 5 punto (este punto es crucial y refleja la priorización de la exposición).

**Fecha de presentación:** 29 de mayo desde las 16:00 hrs.


### 1. Generador de texto con LSTM a nivel de caracteres y tokenización BPE
* **Temas principales:** BPE algorithms, LSTM, FNN (para la capa de salida).
* **Descripción del proyecto:**
    -  Implementa el algoritmo Byte Pair Encoding (BPE) para tokenizar un corpus de texto en subpalabras.
    -  Entrena una red LSTM a nivel de caracteres (o subpalabras BPE) para generar texto.
    -  Utiliza una FNN como capa final para predecir el siguiente elemento.
* **Entregables clave (cuaderno de resultados):**
    * Código de la implementación de BPE y el modelo LSTM.
    * Pruebas de tokenización BPE y generación de texto.
    * Benchmarking: Tiempo de entrenamiento por época, velocidad de generación.
    * Profiling: Uso de memoria, identificación de cuellos de botella.
    * Visualización de la curva de pérdida y ejemplos de texto generado.
* **Métricas para la exposición:**
    * Claridad al explicar BPE y su impacto en el vocabulario.
    * Demostración de la generación de texto y análisis de su coherencia.
    * Discusión sobre cómo los hiperparámetros de BPE y LSTM afectan el resultado.
    * Análisis de los resultados de benchmarking y profiling.
* **Desafío y enfoque:** Lograr una generación de texto coherente. La exposición debe destacar cómo BPE maneja palabras desconocidas y la dinámica de aprendizaje de la LSTM.

### 2. Clasificador de sentimiento con embeddings pre-entrenados (Mikolov) y red neuronal
* **Temas principales:** Skip-gram (uso de embeddings pre-entrenados), FNN o RNN/LSTM.
* **Descripción del proyecto:**
    -  Carga embeddings de palabras pre-entrenados (ej. Word2Vec de Mikolov).
    -  Implementa un clasificador de sentimiento (positivo/negativo) sobre un dataset de reviews.
    -  Compara el rendimiento usando una FNN simple vs. una RNN/LSTM básica.
* **Entregables clave (cuaderno de resultados):**
    * Código para cargar embeddings y los modelos de clasificación.
    * Pruebas de clasificación y matriz de confusión.
    * Benchmarking: Tiempo de entrenamiento y predicción para FNN vs. RNN/LSTM.
    * Profiling: Identificar diferencias de recursos entre FNN y RNN/LSTM.
    * Visualización de la precisión/pérdida durante el entrenamiento.
* **Métricas para la exposición:**
    * Explicación del beneficio de usar embeddings pre-entrenados.
    * Comparación clara del rendimiento y complejidad entre FNN y RNN/LSTM para esta tarea.
    * Análisis de errores de clasificación.
    * Discusión sobre los resultados del benchmarking.
* **Desafío y enfoque:** Integrar correctamente los embeddings y analizar las diferencias entre arquitecturas. La exposición debe centrarse en el impacto de los embeddings y la elección del modelo.

### 3. Traductor automático básico (Encoder-Decoder) con RNNs/LSTMs y atención Bahdanau
* **Temas principales:** RNN/LSTM, general attentions (Bahdanau).
* **Descripción del proyecto:**
    -  Implementa un modelo sequence-to-sequence (encoder-decoder) simplificado usando RNNs o LSTMs.
    -  Añade un mecanismo de atención (estilo Bahdanau) al decodificador.
    -  Entrénalo en una tarea muy simple (ej. traducción de frases cortas en un vocabulario limitado, o conversión de formato de fechas).
* **Entregables clave (cuaderno de resultados):**
    * Código del modelo Seq2Seq con atención.
    * Pruebas de traducción en un conjunto de validación pequeño.
    * Benchmarking: Tiempo de entrenamiento por época.
    * Profiling: Identificar la carga computacional de la atención.
    * Visualización de los pesos de atención para algunas secuencias de entrada/salida.
* **Métricas para la exposición:**
    * Explicación clara de la arquitectura encoder-decoder y el mecanismo de atención.
    * Demostración de ejemplos de traducción y cómo la atención ayuda a alinear palabras.
    * Análisis de la visualización de los pesos de atención.
    * Discusión sobre las limitaciones del modelo simplificado.
* **Desafío y enfoque:** La complejidad de implementar correctamente Seq2Seq y atención en poco tiempo. La exposición debe enfocarse en el rol de la atención y su visualización.


### 4.  Detector de anomalías en secuencias de texto usando autoencoders basados en RNN
* **Temas principales:** Autoencoders (AEs), RNN/LSTM (como componentes del AE).
* **Descripción del proyecto:**
    -  Construye un autoencoder donde el encoder y el decoder sean RNNs o LSTMs.
    -  Entrena el autoencoder en un corpus de secuencias de texto "normales".
    -  Utiliza el error de reconstrucción para identificar secuencias de texto anómalas o atípicas.
* **Entregables clave (cuaderno de resultados):**
    * Código del autoencoder RNN/LSTM.
    * Pruebas mostrando el error de reconstrucción para secuencias normales vs. anómalas.
    * Benchmarking: Tiempo de entrenamiento y de inferencia para calcular el error.
    * Profiling: Uso de memoria del modelo.
    * Visualización de la distribución de errores de reconstrucción.
* **Métricas para la exposición:**
    * Explicación del concepto de autoencoder para detección de anomalías.
    * Demostración con ejemplos claros de secuencias normales y anómalas y sus errores.
    * Discusión sobre cómo se define un umbral para la anomalía.
    * Análisis de la efectividad del modelo.
* **Desafío y enfoque:** Definir y encontrar datos "anómalos" y ajustar el AE. La exposición debe mostrar cómo el error de reconstrucción es un indicador útil.


### 5. Implementación de Skip-gram desde cero con visualización de embeddings
* **Temas principales:** Skip-gram (Mikolov), FNN (como parte de la red superficial de Skip-gram).
* **Descripción del proyecto:**
    -  Implementa el modelo Skip-gram (con negative sampling o softmax jerárquico simplificado) para aprender embeddings de palabras.
    -  Entrénalo en un corpus de texto.
    -  Visualiza los embeddings aprendidos usando t-SNE o PCA para mostrar relaciones semánticas.
* **Entregables clave (cuaderno de resultados):**
    * Código de la implementación de Skip-gram.
    * Pruebas de similitud de palabras (ej. `rey - hombre + mujer = reina`).
    * Benchmarking: Tiempo de entrenamiento.
    * Profiling: Optimización del muestreo negativo si se implementa.
    * Visualización 2D/3D de los embeddings.
* **Métricas para la exposición:**
    * Explicación intuitiva del funcionamiento de Skip-gram y el objetivo de aprendizaje.
    * Demostración de las relaciones capturadas en los embeddings (analogías, clusters).
    * Análisis de la visualización: ¿qué palabras aparecen juntas?
    * Discusión sobre los parámetros clave del modelo.
* **Desafío y enfoque:** Implementar eficientemente el proceso de entrenamiento de Skip-gram. La exposición debe ser muy visual, mostrando la "magia" de los embeddings.

### 6. Resumidor extractivo de texto con LSTMs y atención Luong (simplificado)
* **Temas principales:** LSTM, general attentions (Luong), FNN (para puntuación).
* **Descripción del proyecto:**
    -  Implementa un modelo simplificado para la sumarización extractiva.
    -  Utiliza LSTMs para codificar frases del documento.
    -  Aplica un mecanismo de atención (estilo Luong) para ponderar la importancia de las frases o palabras clave en relación con el documento completo (o una consulta si la hubiera).
    -  Selecciona las K frases con mayor puntuación.
* **Entregables clave (cuaderno de resultados):**
    * Código del codificador LSTM y el mecanismo de atención/puntuación.
    * Pruebas con ejemplos de resúmenes generados para artículos cortos.
    * Benchmarking: Tiempo para generar un resumen.
    * Profiling: Carga de la atención.
    * Visualización (si es posible) de cómo la atención se enfoca en ciertas partes para la selección.
* **Métricas para la exposición:**
    * Explicación del enfoque extractivo y cómo se puntúan las frases.
    * Demostración de resúmenes generados y su comparación con el original.
    * Análisis de cómo la atención (si se visualiza) contribuye a la selección.
    * Discusión sobre la coherencia y cobertura de los resúmenes.
* **Desafío y enfoque:** La exposición debe centrarse en la lógica del modelo y los resultados cualitativos.


### 7. "Mini" neural Turing machine (NTM) para una tarea algorítmica simple
* **Temas principales:** Neural Turing Machines (NTMs), LSTM (como controlador), FNN (para cabezales de lectura/escritura).
* **Descripción del proyecto:**
    -  Implementa una versión muy simplificada de una NTM. Enfócate en los componentes clave: controlador (LSTM), memoria externa y cabezales de lectura/escritura con mecanismos de direccionamiento básicos (basado en contenido o ubicación).
    -  Intenta entrenarla en una tarea algorítmica muy simple, como la tarea de copia de secuencias (copiar una secuencia de entrada a la salida).
* **Entregables clave (cuaderno de resultados):**
    * Código de los componentes clave de la NTM (controlador, memoria, cabezales, direccionamiento).
    * Pruebas en la tarea de copia (ej. mostrar la precisión en secuencias cortas).
    * Benchmarking: Tiempo de entrenamiento por época (probablemente será lento).
    * Profiling: Identificar las partes más costosas computacionalmente.
    * Visualización (si es posible y el tiempo lo permite): Patrones de acceso a la memoria, pesos de los cabezales.
* **Métricas para la exposición:**
    * Explicación clara de la arquitectura de una NTM y la función de cada componente.
    * Demostración (incluso si es con resultados parciales) del aprendizaje en la tarea de copia.
    * Discusión sobre los desafíos de implementación y entrenamiento de NTMs.
    * Análisis de las visualizaciones (si se logran).
* **Desafío y enfoque:** La complejidad inherente de las NTMs. La exposición debe centrarse en la comprensión conceptual y los desafíos, más que en un rendimiento perfecto.


### 8. Exploración de un componente de differentiable neural computer (DNC)
* **Temas principales:** Differentiable Neural Computers (DNCs), LSTM (como controlador), Attention (para direccionamiento de memoria).
* **Descripción del proyecto:**
    -  Dado que un DNC completo es muy complejo, enfócate en implementar y probar un componente específico, como:
        * El mecanismo de direccionamiento de memoria (content-based y allocation weighting).
        * Los enlaces temporales entre posiciones de memoria escritas consecutivamente.
    -  Diseña cómo este componente interactuaría con el resto de un DNC conceptual.
    -  Prueba el componente de forma aislada con datos simulados.
* **Entregables clave (cuaderno de resultados):**
    * Código del componente del DNC implementado.
    * Pruebas unitarias o funcionales del componente.
    * Benchmarking: Tiempo de ejecución del componente.
    * Profiling: Complejidad del componente.
    * Diagramas o explicaciones de cómo este componente encaja en un DNC completo.
* **Métricas para la exposición:**
    * Explicación clara de los conceptos clave de los DNCs (memoria externa, pesos de uso, enlaces temporales) y cómo se diferencian de las NTMs.
    * Descripción detallada del componente implementado y su función.
    * Demostración del funcionamiento del componente.
    * Discusión sobre el potencial de los DNCs para tareas de razonamiento.
* **Desafío y enfoque:** La altísima complejidad de los DNCs. Al igual que con la NTM, el objetivo es una comprensión profunda de una parte.
  La exposición debe educar sobre la arquitectura DNC a través del componente elegido. 

