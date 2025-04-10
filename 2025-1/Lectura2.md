#### 1. Definición de modelos de lenguaje

Los modelos de lenguaje son sistemas de inteligencia artificial capaces de procesar y generar texto de manera coherente y contextualizada. Estos modelos aprenden distribuciones de probabilidad sobre secuencias de palabras, es decir, se entrenan para predecir la siguiente palabra (o token) dado un contexto previo. A lo largo del tiempo, la comunidad de investigación en procesamiento de lenguaje natural (NLP) ha desarrollado distintos enfoques para la creación y el perfeccionamiento de los modelos de lenguaje, haciendo uso de enormes cantidades de datos textuales.

En su evolución histórica, se ha pasado de sistemas basados en reglas —que utilizaban conocimiento lingüístico explícitamente codificado— a métodos estadísticos, para luego llegar al aprendizaje profundo (deep learning). El uso de redes neuronales profundas y arquitecturas avanzadas, como los transformers, ha permitido la construcción de modelos de gran escala conocidos como Grandes Modelos de Lenguaje (LLMs, por sus siglas en inglés). Estos LLM cuentan con miles de millones de parámetros y están entrenados sobre corpus inmensos, llegando a manejar contextos cada vez más amplios y a generar texto de notable coherencia.

Dentro de esta categoría de modelos, se pueden mencionar ejemplos como GPT (Generative Pre-Trained Transformer), BERT (Bidirectional Encoder Representations from Transformers), BART (Bidirectional and Auto-Regressive Transformers) y T5 (Text-to-Text Transfer Transformer). Todos ellos aprovechan el concepto de embeddings, representaciones numéricas densas que capturan relaciones semánticas y sintácticas entre palabras. 
También usan mecanismos de autoatención (self-attention) que permiten identificar qué partes de la secuencia son más relevantes al generar o interpretar texto.

La utilidad de estos modelos se extiende a múltiples aplicaciones de procesamiento del lenguaje, tales como:
- Traducción automática.
- Análisis de sentimientos.
- Resúmenes automáticos.
- Chatbots que mantienen conversaciones contextualmente apropiadas.
- Respuestas a preguntas (question answering).
- Etiquetado de secuencias y extracción de información.

Estos grandes modelos de lenguaje requieren un entrenamiento cuidadoso y un conjunto de datos masivo para capturar las diversas estructuras del lenguaje. Además, se utilizan técnicas de ajuste fino (fine-tuning) para tareas específicas, aprovechando el conocimiento general aprendido en la fase de preentrenamiento.


#### 2. Evaluación de modelos

La evaluación de modelos de inteligencia artificial, especialmente en el ámbito del procesamiento del lenguaje natural, es un paso crucial para determinar su calidad y desempeño. No basta con construir un modelo y comprobar que produce una salida plausible; se requiere medir, mediante métricas cuantitativas y cualitativas, la correspondencia entre las respuestas generadas y los objetivos o etiquetas de referencia.

En la IA generativa, la evaluación puede volverse más compleja que en otras ramas del aprendizaje automático. Esto se debe a que el modelo produce contenido “creativo” o “novedoso” que, en muchas ocasiones, no cuenta con una única referencia correcta. Aun así, los investigadores han desarrollado distintas herramientas para determinar hasta qué punto el contenido generado se aproxima a un estándar o se considera útil y coherente.

En el caso de generación de texto —como la traducción automática, la generación de resúmenes o la creación de respuestas en un chatbot—, se suelen aplicar métricas automáticas que comparan el texto generado con uno o varios textos de referencia (o gold standard). Estas métricas buscan medir la superposición en términos de palabras, frases o n-grams. Es importante destacar que la evaluación puramente automática puede ser incompleta, ya que no necesariamente capta la calidad estilística, factual o creativa de los textos generados, pero sí aporta criterios objetivos para guiar el entrenamiento y la selección de hiperparámetros.

Además de las métricas cuantitativas (como BLEU, ROUGE, METEOR, perplejidad, etc.), los investigadores a menudo recurren a la evaluación humana para complementar la valoración, especialmente cuando es fundamental asegurar que el texto sea coherente, veraz y apropiado para usuarios finales. Estas evaluaciones humanas pueden implicar cuestionarios, pruebas de preferencia o escalas de calidad lingüística.

#### 3. Principales arquitecturas y modelos subyacentes

La IA generativa abarca diversas arquitecturas y metodologías, cada una adecuada para distintos tipos de datos y tareas:

1. **Redes neuronales recurrentes (RNNs)**  
   Son apropiadas para datos secuenciales o series temporales. Se caracterizan por un bucle interno que permite al modelo "recordar" información de pasos anteriores y usarla para predecir la siguiente salida. Sin embargo, tienden a tener dificultades para mantener dependencias largas.

2. **Transformers**  
   Emplean un mecanismo de autoatención (self-attention) que permite asignar un peso relativo a cada parte de la secuencia. Estas arquitecturas han supuesto un gran avance en NLP, pues facilitan procesar secuencias en paralelo y capturar relaciones de largo alcance sin los problemas de memoria que presentan las RNN.

3. **Redes generativas antagónicas (GANs)**  
   Están formadas por dos redes: un generador y un discriminador que compiten entre sí. El generador intenta crear muestras que parezcan reales, mientras que el discriminador trata de distinguir entre muestras auténticas y generadas. Esta competencia conduce a una mejora progresiva de la calidad de los datos generados.

4. **Autoencoders variacionales (VAEs)**  
   Trabajan con una fase de codificación y otra de decodificación, mapeando la entrada a una distribución latente y usando luego muestreos de esa distribución para decodificar a una salida. Son muy útiles para la generación de datos con variaciones controladas.

5. **Modelos de difusión (Diffusion Models)**  
   Se entrenan para remover ruido de imágenes o de otros tipos de datos, iterativamente, hasta obtener una muestra generada coherente. Basan su eficacia en el aprendizaje de estadísticos y la reconstrucción de ejemplos distorsionados.


#### 4. Evolución histórica de la IA generativa

La historia de la IA generativa puede describirse en diferentes fases:

- **Sistemas basados en reglas**: Dependían de reglas lingüísticas o simbólicas codificadas a mano. Dada su rigidez, no podían aprender de nuevos datos ni generalizar fuera de lo establecido.

- **Métodos estadísticos y aprendizaje automático**: Introdujeron un enfoque más flexible y adaptativo, permitiendo estimar distribuciones y probabilidades a partir de grandes conjuntos de datos. Este paso resultó esencial para el futuro desarrollo de técnicas más complejas.

- **Deep Learning y redes neuronales**: El crecimiento en capacidad de cómputo, sumado a grandes bases de datos, impulsó la adopción masiva de redes neuronales profundas. Modelos como las RNN y sus variantes (LSTM, GRU) permitieron avances importantes en generación de lenguaje.

- **Transformers**: Son la culminación más reciente de esta evolución, ya que introducen un modelo que supera limitaciones de secuencialidad y captura mejor las dependencias largas. Esto ha resultado en grandes mejoras en traducción automática, respuesta a preguntas, generación de texto, resúmenes, clasificación y otras tareas.

#### 5. Perplejidad BLEU

En el marco de la evaluación de modelos de lenguaje y de sus variantes generativas, aparecen varios indicadores para medir la calidad del texto producido. Dos métricas especialmente relevantes son la perplejidad y el BLEU, cada una con un propósito distinto:

**Perplejidad**  
- Se utiliza comúnmente para medir qué tan bien un modelo de lenguaje predice una secuencia de palabras. Cuanto menor sea la perplejidad, mejor será la capacidad del modelo para anticipar el próximo token dentro de un contexto.  
- Conceptualmente está asociada a la entropía: una alta perplejidad indica que el modelo está "confundido" al predecir la siguiente palabra, mientras que una perplejidad baja sugiere mayor certidumbre y mejor ajuste a las regularidades lingüísticas.

**BLEU (Bilingual Evaluation Understudy)**  
- Es una de las métricas pioneras y más utilizadas en la evaluación de la calidad de la traducción automática.  
- Compara el texto generado (hipótesis) con uno o varios textos de referencia, contando la superposición de n-gramas.  
- Aplica el concepto de "clipped precision" para evitar premiar repeticiones excesivas de n-grams que aparezcan en la referencia.  
- Incluye también una penalización por brevedad, de modo que no se valore con alta puntuación una traducción mucho más corta que la referencia.  
- Un BLEU alto indica que el texto generado se acerca significativamente a la o las referencias utilizadas.

Estas dos métricas abarcan diferentes aspectos de evaluación. La perplejidad se enfoca más en la capacidad predictiva interna del modelo, en su coherencia con la distribución del lenguaje, mientras que BLEU se orienta a medir la similitud n-gram entre un texto generado y uno de referencia (típicamente en traducción). Ambas herramientas son de gran ayuda para los investigadores, aunque no siempre reflejan la totalidad de la "calidad" de un texto según criterios de estilo, veracidad o fluidez subjetiva.


#### 6. ROUGE y BLEU

ROUGE y BLEU a menudo se discuten de forma conjunta en la evaluación de modelos de generación de texto, ya que ambos se basan en mediciones de superposición de n-grams (entre otros recursos), pero se aplican en diferentes contextos:

1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**  
   - Se utiliza principalmente para tareas de resumen automático.  
   - Mide la superposición entre el resumen generado y uno o más resúmenes de referencia.  
   - **ROUGE-N** cuenta los n-grams comunes.  
   - **ROUGE-L** se basa en la longitud de la subsecuencia común más larga (LCS) entre la hipótesis y la referencia.  
   - **ROUGE-S** considera skip-grams, permitiendo cierta flexibilidad en la coincidencia.  
   - Un valor de ROUGE alto indica que el contenido esencial del resumen de referencia se ha capturado en gran medida.

2. **BLEU (Bilingual Evaluation Understudy)**  
   - Si bien se mencionó también en la sección anterior (relacionada con perplejidad), vuelve a ser relevante aquí como comparativa con ROUGE.  
   - Su principal uso histórico es para traducción automática, estimando la precisión en la coincidencia de n-grams.  
   - Las versiones originales de BLEU hacen énfasis en la precisión (n-gramas generados que coinciden con referencia), mientras que ROUGE normalmente se centra más en el recall (n-grams de referencia que aparecen en la generación).  

Ambas métricas pueden emplearse en más de un tipo de tarea, aunque en la práctica su uso convencional es:
- **ROUGE**: Resumen (y a veces otras tareas donde el contenido clave debe coincidir).  
- **BLEU**: Traducción automática (u otras formas de generación donde se desea medir una precisión de solapamiento con la referencia).

Cabe anotar que ninguna de estas métricas capta por completo la fluidez, creatividad o corrección factual del texto. Por ello, en muchos casos se combinan con METEOR, evaluación humana o pruebas específicas diseñadas para la tarea. En el caso de la traducción, por ejemplo, la evaluación humana puede ser esencial para percibir matices idiomáticos o identificar errores de significado que no se reflejan necesariamente en la coincidencia de n-grams.


#### 7. Preparación de datos en NLP: Tokenización y carga de datos

La preparación de los datos es un paso fundamental en cualquier proyecto de inteligencia artificial, y aún más cuando se aborda el procesamiento del lenguaje natural. Dos de los aspectos más relevantes son:

1. **Tokenización**  
   - Consiste en dividir el texto en unidades mínimas (tokens). Pueden ser palabras, caracteres o subpalabras.  
   - **Tokenización basado en palabras**: Mantiene las palabras completas, lo que a menudo preserva el significado semántico, pero incrementa de forma notable el tamaño del vocabulario.  
   - **Tokenización basado en caracteres**: Reduce el vocabulario, pero puede perder nociones semánticas.  
   - **Tokenización basado en subpalabras**: Combina lo mejor de las dos aproximaciones anteriores. Frecuentemente utilizadas en modelos de vanguardia (WordPiece, Unigram, SentencePiece), pues mantienen intactas las palabras frecuentes y dividen solo aquellas poco comunes.  
   - Se añaden tokens especiales como `<bos>` (beginning of sentence) y `<eos>` (end of sentence) para indicar límites de secuencia.

2. **Data sets y Data Loaders en PyTorch**  
   - **Data sets**: Objetos que representan colecciones de muestras (input features y etiquetas).  
   - **Data Loader**: Un mecanismo que organiza y entrega los datos por lotes (batches), permitiendo entrenar eficientemente en cada época. También facilita el uso de múltiples hilos para lectura, la aleatorización (shuffling) y otras operaciones de preprocesamiento.


#### 8. Representaciones del texto: One-hot encoding, Bag-of-Words y Embeddings

Para que un modelo trabaje con texto, es necesario convertir las palabras en vectores numéricos. Existen varios métodos:

1. **One-hot encoding**  
   - Cada palabra se codifica como un vector de longitud igual al tamaño del vocabulario, con un "1" en la posición correspondiente a esa palabra y "0" en el resto.  
   - Puede ser muy grande (si el vocabulario es extenso) y no refleja relaciones semánticas entre palabras.

2. **Bag-of-Words (BoW)**  
   - Representa un documento contando (o promediando) los vectores one-hot de las palabras que aparecen, sin tener en cuenta el orden.  
   - A pesar de su simplicidad, puede usarse con éxito en tareas como clasificación de textos, aunque pierde información de secuencia.

3. **Embeddings**  
   - Son representaciones densas y de menor dimensión que capturan relaciones semánticas y sintácticas.  
   - En PyTorch se implementan con las clases `Embedding` y `EmbeddingBag`.  
   - Los embeddings permiten que palabras similares (semántica o contextualmente) estén cerca en el espacio vectorial.


#### 9. Redes neuronales y clasificación de documentos

Con el texto tokenizado y sus representaciones vectoriales listas, se puede construir un clasificador de documentos basado en redes neuronales. Algunos puntos clave:

- **Secuencia de capas**: Una red neuronal se compone de múltiples capas (al menos de entrada y de salida, pudiendo incluir capas ocultas). Cada capa aplica transformaciones lineales y, normalmente, funciones de activación no lineales.  
- **Argmax para clasificación**: La red suele producir un vector de logits (una puntuación por cada categoría). La clase predicha corresponde al índice con valor máximo.  
- **Hiperparámetros**: Tasa de aprendizaje, número de épocas, tamaño del batch, número de capas, funciones de activación, etc. Se ajustan según la tarea y los datos para encontrar la mejor configuración.  
- **Función de predicción**: Una vez entrenado el modelo, se aplica la tokenización al texto de entrada y se pasa al modelo, que devuelve la etiqueta con mayor probabilidad.

#### 10. Entrenamiento de redes neuronales y proceso de optimización

El entrenamiento de una red neuronal busca ajustar los pesos (parámetros aprendibles) para minimizar la función de pérdida:

1. **Parámetros aprendibles**  
   - Son los valores (matrices, vectores) que la red usa en sus transformaciones internas.

2. **Función de pérdida (Loss function)**  
   - En clasificación multi-clase, la entropía cruzada (Cross-Entropy) es muy utilizada, puesto que evalúa cuán cerca están las distribuciones de probabilidad predicha y la real.

3. **Técnica de Monte Carlo**  
   - Para estimar la esperanza de la pérdida sobre todo el conjunto de datos, se trabaja por lotes (mini-batches), calculando gradientes en cada paso para actualizar los pesos.

4. **Optimización**  
   - Se emplean algoritmos como SGD, Adam, RMSProp, etc. La finalidad es encontrar los parámetros que minimicen la pérdida.  
   - Se acostumbra dividir los datos en tres subconjuntos: entrenamiento (para aprender), validación (para ajustar hiperparámetros) y prueba (para medir el rendimiento final).

5. **Ciclo de entrenamiento**  
   - Para cada época, se recorre el conjunto de entrenamiento en lotes.  
   - Se calcula la pérdida y se retropropaga el error para actualizar los parámetros.  
   - Se registra la evolución de la pérdida y otras métricas para monitorear el sobreajuste o el subajuste.

#### 11. Modelos n-grama y su extensión en redes neuronales

Los modelos estadísticos basados en n-gramas son una aproximación previa al deep learning, pero siguen siendo conceptualmente importantes:

- **Bi-grama**: La palabra siguiente depende solo de la anterior (contexto de 1).  
- **Tri-grama**: La palabra siguiente depende de las dos anteriores (contexto de 2).  
- **N-grama general**: Permite un contexto más amplio. Para grandes `n`, el número de posibles secuencias crece exponencialmente, lo que hace necesario buscar formas de generalización.

En redes neuronales, se puede simular un modelo n-grama construyendo un vector de contexto concatenando los embeddings de las `n` palabras previas y usando una capa adicional para producir la palabra siguiente. 
Este procedimiento se desplaza a lo largo de la secuencia (ventana deslizante) para entrenar el modelo de forma integral.

#### 12. Word2Vec, CBOW y Skip-gram

**Word2Vec** es un conjunto de técnicas para aprender embeddings de palabras. Sus dos variantes principales son:

1. **Continuous Bag of Words (CBOW)**  
   - Dadas las palabras de contexto, se predice la palabra central.  
   - El embedding de la palabra objetivo se construye a partir de la combinación (normalmente la suma o promedio) de los embeddings de sus contextos.

2. **Skip-gram**  
   - Tarea inversa: dada la palabra central, se predicen las palabras de contexto.  
   - Se entrena de forma que cada muestra considere la palabra objetivo y una de sus vecinas, generando más instancias de entrenamiento.

Este método de embeddings ha demostrado su eficacia capturando relaciones semánticas, por ejemplo, sinónimos o relaciones como rey - hombre + mujer ≈ reina. Además de Word2Vec, existen otras propuestas como GloVe (Global Vectors) y fastText, igualmente utilizadas para obtener representaciones vectoriales.


#### 13. Modelos secuencia a secuencia (sequence-to-sequence) con RNN

Cuando se requiere transformar una secuencia en otra (traducción, subtitulado automático, etc.), los modelos secuencia a secuencia (seq2seq) son adecuados. A menudo se basan en RNNs, LSTMs o GRUs:

- **Arquitectura encoder-decoder**  
  - **Encoder**: Recibe la secuencia de entrada, token por token, y condensa la información en un estado oculto final (contexto).  
  - **Decoder**: Usa ese estado para generar la secuencia de salida de forma autoregresiva, produciendo un token a la vez.

- **Dificultades de entrenamiento**  
  - Las RNN son propensas al problema de gradiente desaparecido o explosivo, especialmente con secuencias largas.  
  - Mecanismos de atención (attention) ayudan a mitigar este problema al permitir que el decoder "observe" selectivamente distintas partes de la secuencia de entrada.


#### 14. Métricas de evaluación en modelos generativos de texto

La generación de texto plantea el desafío de evaluar la calidad de la salida de manera objetiva. Varias métricas se han popularizado:

- **Perplejidad**  
  Ya mencionada, indica cuán bien el modelo predice la secuencia. Se asocia a la entropía: cuanto menor, mejor es la capacidad de generalización.

- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**  
  Enfocado en la evaluación de resúmenes. Compara n-grams (ROUGE-N), subsecuencias más largas (ROUGE-L) y skip-grams (ROUGE-S) entre hipótesis y referencia.

- **BLEU (Bilingual Evaluation Understudy)**  
  Mide la coincidencia n-gram en tareas de traducción, aplicando precisión recortada y penalización por brevedad.

- **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**  
  Considera la precisión y la exhaustividad (recall), así como la alineación entre palabras, manejo de sinónimos y otros aspectos no contemplados en BLEU.

Todas ellas, sin embargo, pueden complementar —pero no sustituir plenamente— la evaluación humana, especialmente cuando se requiere juzgar la coherencia global, la veracidad factual o la creatividad.


#### 15. Consideraciones éticas y sesgos en los embeddings

El uso de grandes conjuntos de datos para entrenar modelos de lenguaje conlleva riesgos de incorporar sesgos y de exponer información sensible:

1. **Sesgos en el lenguaje**  
   - Modelos como Word2Vec pueden reflejar asociaciones estereotipadas (por ejemplo, vincular ciertas profesiones a un género).  
   - Esto puede afectar las recomendaciones o decisiones que tomen sistemas automatizados.

2. **Técnicas de debiasing**  
   - Han surgido métodos para detectar y reducir el sesgo en los vectores de embeddings, realineando o neutralizando dimensiones relacionadas con atributos sensibles (género, raza, etc.).

3. **Privacidad y datos**  
   - Es posible que los modelos memoricen información personal si el corpus de entrenamiento incluye datos sensibles.  
   - Se aplican técnicas como la anonimización y la privacidad diferencial para reducir el riesgo de divulgación de datos concretos.

4. **Representación justa**  
   - Los modelos deben entrenarse con datos diversos y equilibrados para no discriminar a minorías o a grupos específicos.

#### 16. Entrenamiento y evaluación de modelos secuencia a secuencia

Para entrenar un modelo seq2seq (o de traducción automática) y evaluar su rendimiento, se sigue un proceso similar al de cualquier red neuronal, con particular énfasis en:

1. **Inicialización del modelo en modo de entrenamiento**  
   - Se activan capas como dropout, esenciales para regularizar y mejorar la generalización.

2. **Ciclo de iteraciones**  
   - Se toman lotes de datos (cada lote con secuencias de entrada y su correspondiente salida).  
   - El decoder produce la traducción u otro tipo de secuencia destino token a token.  
   - Se calcula la pérdida (por ejemplo, entropía cruzada) comparando el resultado con la secuencia real.

3. **Perplejidad**  
   - Para modelos seq2seq, la perplejidad se aplica como métrica agregada para medir la calidad general de la predicción.

4. **Traducción (inferencia)**  
   - En la etapa de inferencia, el decoder genera tokens uno por uno, usando el token predicho anteriormente como entrada para el siguiente paso.  
   - Se puede usar búsqueda heurística (beam search, greedy decoding) para encontrar la secuencia de salida que maximice la probabilidad del modelo.

>Observación: La métrica de perplexidad se emplea con frecuencia para evaluar la eficiencia de los grandes modelos de lenguaje (LLMs) y de la IA generativa.
>Este indicador cuantifica cuánto "sorpresa" o "incertidumbre" muestra el modelo al predecir la siguiente palabra en una secuencia. Cuanto más bajo sea el valor de perplexidad, mayor es la certidumbre y, por ende, mejor el rendimiento del modelo al generar texto. Una perplexidad reducida indica que el modelo ha aprendido adecuadamente las estructuras y dependencias del lenguaje. Así, esta métrica resulta esencial para contrastar diferentes arquitecturas y detectar oportunidades de optimización en el ámbito del procesamiento de lenguaje natural.
