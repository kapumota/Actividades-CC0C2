### Representaciones distribuidas

Las representaciones distribuidas constituyen hoy en día la piedra angular de los avances más significativos en procesamiento de lenguaje natural (NLP) y aprendizaje automático. Frente a los métodos de codificación **local**, por ejemplo vectores dispersos donde cada palabra se identifica con una dimensión dedicada, 
las representaciones distribuidas modelan palabras, frases u oraciones como puntos en un espacio continuo de dimensiones moderadas (decenas o cientos). 

Este cambio de paradigma nace de la hipótesis distributiva, que plantea que unidades lingüísticas con contextos de uso similares comparten  aspectos de significado y función. Gracias a ello, se logra no solo capturar sinónimos y analogías, sino también dotar a los sistemas de la
capacidad de inferir representaciones de términos infrecuentes o incluso desconocidos. 

A su vez, esta premisa facilita el manejo de vocabularios masivos sin incurrir en costosos recursos de almacenamiento, al mantener los vectores de longitud constante y densos. 

### Antecedentes de las representaciones distribuidas 

#### Hipótesis distributiva y fundamentos lingüísticos 

El origen teórico de las representaciones distribuidas se remonta a observaciones en lingüística de mediados del siglo XX, cuando se expresó  
que "You shall know a word by the company it keeps" (J. R. Firth). Bajo esta premisa se postuló que el significado de una palabra se define en gran medida por sus contextos de uso, es decir, por las palabras que la rodean en textos reales. Con la disponibilidad de grandes corpora digitalizados, surgió la posibilidad computacional de cuantificar coocurrencias y patrones de contexto, sentando las bases de la semántica distribuida. 

#### Primeros enfoques basados en conteo 

Durante décadas, los pioneros se apoyaron en matrices de coocurrencia: tablas que contabilizan cuántas veces aparece cada par de palabras en una ventana de tamaño fijo. A partir de ahí, tecnologías como TF–IDF permitieron ponderar términos según su importancia relativa, y métodos de descomposición de matrices como análisis semántico latente (LSA) tradujeron la información cruda 
en vectores de menor dimensión. 
Exploraciones posteriores aplicaron transformaciones basadas en información mutua o probabilidades conjuntas para rescatar relaciones implícitas. 

Aunque efectivos, estos enfoques sufrían de escasa capacidad de generalización ante vocabularios en constante crecimiento y de limitaciones para capturar relaciones no lineales. 

#### Ejemplo

#### 1. Corpus de ejemplo y definición de ventana

Supongamos un mini‑corpus de **3 frases** y una ventana de tamaño 1 (solo palabras adyacentes):

1. “El **perro** **persigue** al **gato**.”  
2. “El **gato** **trepa** al **árbol**.”  
3. “El **perro** ve al **árbol**.”

Vocabulario (orden fijo):  
```
[ perro, persigue, gato, trepa, árbol ]
```

#### 2. Matriz de coocurrencia

Cada celda `(i,j)` cuenta cuántas veces `términoᵢ` aparece junto a `términoⱼ` en la ventana. La matriz es simétrica:

|             | perro | persigue | gato | trepa | árbol |
|-------------|:-----:|:--------:|:----:|:-----:|:-----:|
| **perro**   |   –   |     1    |   0  |   0   |   1   |
| **persigue**|   1   |     –    |   1  |   0   |   0   |
| **gato**    |   0   |     1    |   –  |   1   |   0   |
| **trepa**   |   0   |     0    |   1  |   –   |   1   |
| **árbol**   |   1   |     0    |   0  |   1   |   –   |

- **perro–persigue**: 1 vez (frase 1)  
- **persigue–gato**: 1 vez (frase 1)  
- **gato–trepa**: 1 vez (frase 2)  
- **trepa–árbol**: 1 vez (frase 2)  
- **perro–árbol**: 1 vez (frase 3)

#### 3. Ponderación TF–IDF (esquema simplificado)

Para capturar la "importancia" de cada término, se suele aplicar TF–IDF sobre la matriz de conteos:

- **TF**: en este ejemplo cada par aparece a lo sumo 1 vez, así que TF=1.  
- **IDF** (_inverso de documentos que contienen el término_):  
  - perro aparece en 2/3 frases → IDF≈log(3/2)=0.18  
  - persigue en 1/3 → IDF≈log(3/1)=1.10  
  - gato en 2/3 → IDF≈0.18  
  - trepa en 1/3 → IDF≈1.10  
  - árbol en 2/3 → IDF≈0.18  

Multiplicando, obtenemos la matriz ponderada (redondeado):

|             | perro | persigue | gato | trepa | árbol |
|-------------|:-----:|:--------:|:----:|:-----:|:-----:|
| **perro**   |   –   | 1·1.10=1.10 | 0 |   0   | 1·0.18=0.18 |
| **persigue**|1·0.18=0.18|   –    |1·0.18=0.18| 0 |   0   |
| **gato**    |   0   | 1·1.10=1.10 |   –  |1·1.10=1.10| 0 |
| **trepa**   |   0   |     0    |1·0.18=0.18|   –   |1·0.18=0.18|
| **árbol**   |0.18|     0    |   0  |0.18|   –   |


#### 4. Análisis semántico latente (LSA)

1. **Formulación**: aplicamos descomposición en valores singulares (SVD) a la matriz TF–IDF  
2. **Truncamiento**: nos quedamos con los 2 valores singulares más grandes (σ₁, σ₂) y sus vectores asociados  
3. **Espacio latente**: cada palabra queda representada como un vector de dimensión 2

**Valores singulares (ejemplo)**

| Componente | σ |
|------------|---|
| σ₁         | 2.30 |
| σ₂         | 1.15 |

**Vectores en dimensión 2 (aproximados)**

| Término   |  D1  |  D2  |
|-----------|:----:|:----:|
| perro     | 1.45 | 0.23 |
| persigue  | 0.98 | 0.60 |
| gato      | 1.40 | 0.18 |
| trepa     | 0.15 | 1.05 |
| árbol     | 0.30 | 1.20 |


#### 5. Interpretación

- **Palabras cercanas** en el espacio latente comparten contexto:  
  - *perro* (1.45, 0.23) y *gato* (1.40, 0.18) quedan muy próximos → ambos aparecen junto a verbos de "acción"  
  - *trepa* y *árbol* quedan emparejados en la segunda dimensión → reflejando la relación "gato trepa árbol" y "perro ve árbol"  
- **Generalización**: al reducir ruido y retener solo las componentes principales, LSA mitiga la esparsidad y permite capturar sinonimias parciales.

Así, partiendo de simples conteos de coocurrencia, TF–IDF y SVD nos ofrecen un **espacio semántico continuo** donde medir similitudes y diferencias de forma más robusta que la simple matriz original.

### Modelos predictivos de embeddings de palabras 

#### Arquitectura general y objetivos de entrenamiento 

La siguiente generación de representaciones distribuidas surgió con modelos neuronales diseñados para predecir palabras en contextos supervisados, entrenando simultáneamente matrices de vectores de entrada y salida. 
En esencia, estos modelos aprenden a ajustar sus parámetros de modo que, al presentar un contexto de ejemplo, asignen alta probabilidad a  la palabra correcta y bajas probabilidades a varias "distractoras". 

#### Continuous Bag of Words (CBOW) 

En el paradigma CBOW, el sistema recibe un conjunto de palabras vecinas alrededor de la posición objetivo. Su tarea consiste en combinar esas representaciones contextuales. usualmente promediándolas, para generar una señal conjunta que sirva de
entrada a una capa que predice la palabra central. Este método resulta muy eficiente computacionalmente y suele brillar en colecciones de texto de gran tamaño, donde abundan las regiones de 
contexto ricas en datos. 

#### Skip‑gram 

Skip‑gram invierte el planteamiento: parte de la palabra central para predecir cada una de las palabras de su ventana de contexto. 
Aunque requiere más esfuerzo de cómputo por la cantidad de predicciones a generar, destaca por producir vectores de mayor calidad para términos infrecuentes, ya que cada aparición del término central contribuye a múltiples objetivos de entrenamiento. 

#### Técnicas de optimización 

Para escalar estas arquitecturas a vocabularios de millones de palabras, se introdujeron dos técnicas clave. La primera es el muestreo negativo, que reformula la función de pérdida como una tarea de diferenciación entre el contexto verdadero y un conjunto de ejemplos "negativos" tomados aleatoriamente. 

La segunda es la estimación de ruido, que aproxima las probabilidades globales con distribuciones de ruido conocidas, aligerando la  exigencia de normalizar contra todo el vocabulario en cada paso de entrenamiento. Ambas estrategias
permitieron entrenar modelos con decenas de miles de millones de tokens en tiempos razonables.

#### Ejemplo

**Continuous Bag of Words (CBOW)**  
Imagina que tu modelo recibe un conjunto de palabras que rodean a la palabra que queremos predecir, como un puzle donde faltan piezas centrales.  
- **Contexto**: "El ___ salta sobre el muro." → vemos "El", "salta", "sobre", "el", "muro".  
- **Tarea**: a partir de ese conjunto, el modelo debe adivinar cuál es la palabra que falta ("gato", "niño", "perro", según el corpus).  
- **Ventaja**: al promediar las pistas contextuales, CBOW aprende rápidamente en textos muy largos, donde hay mucha información alrededor de cada término.

**Skip‑gram**  
Aquí partimos de la pieza central y tratamos de reconstruir su vecindario, como si lanzáramos una red a su alrededor para capturar amigos.  
- **Entrada**: "gato"  
- **Objetivo**: predecir los vecinos que suelen acompañar a "gato" —p. ej., "duerme", "almohada", "ronronea"— cada vez que aparece la palabra central.  
- **Ventaja**: cada aparición de "gato" sirve para varios ejercicios de predicción, lo que refuerza la calidad del vector para palabras que salen poco en el texto.

**Muestreo negativo**  
En lugar de enfrentarnos a todo el vocabulario, comparamos ejemplos buenos contra ejemplos "tramposos".  
- Tomamos un par real, como ("gato", "ronronea"), y lo marcamos como correcto.  
- Creamos varios pares falsos, por ejemplo ("gato", "taza") o ("gato", "avión"), y los marcamos como incorrectos.  
- El modelo aprende a distinguir qué pares sí son plausibles según el texto y cuáles no, sin tener que evaluar todas laspalabras posibles en cada paso.

**Estimación de ruido (Noise‑contrastive estimation)**  
Convertimos la tarea de modelar probabilidades complejas en un juego de clasificación frente a un "ruido" conocido.  
- Definimos una distribución sencilla de palabras frecuentes ("el", "de", "y", etc.) como nuestro ruido.  
- Por cada ejemplo real de contexto, generamos un grupo de palabras muestreadas de esa distribución de ruido.  
- El modelo se entrena para clasificar correctamente si un par viene del texto real o de la distribución de ruido, evitando normalizar sobre millones de términos en cada actualización.

### Embeddings preentrenados y transferencia de conocimiento 

#### Modelos de factorización global: GloVe 

GloVe propone una aproximación híbrida entre conteo y predicción. Parte de una matriz global de coocurrencias y ajusta vectores de palabras 
de modo que las distancias en el espacio vectorial reflejen las razones de coocurrencia. 

En la práctica, su ventaja radica en combinar información local de contexto con métricas globales de corpus, estabilizando la calidad de los vectores resultantes. 

**Ejemplo conceptual de GloVe**  

Imaginemos un mini‑corpus de 5 oraciones y un vocabulario reducido:

1. "El **gato** se sienta sobre la **alfombra**."  
2. "El **perro** duerme en la **alfombra**."  
3. "El **gato** persigue al **ratón**."  
4. "El **ratón** come **queso** en la cocina."  
5. "El **perro** ladra cerca de la **cocina**."

**1. Matriz global de coocurrencias**

Primero contamos, en todo el corpus, cuántas veces aparece cada par de palabras dentro de una ventana (por ejemplo ±2 palabras). El resultado podría resumirse así (valores ficticios):

|             | gato | perro | ratón | alfombra | queso | cocina |
|-------------|:----:|:-----:|:-----:|:--------:|:-----:|:------:|
| **gato**    |  –   |   0   |   1   |     1    |   0   |   0    |
| **perro**   |  0   |   –   |   0   |     1    |   0   |   1    |
| **ratón**   |  1   |   0   |   –   |     0    |   1   |   1    |
| **alfombra**|  1   |   1   |   0   |     –    |   0   |   0    |
| **queso**   |  0   |   0   |   1   |     0    |   –   |   1    |
| **cocina**  |  0   |   1   |   1   |     0    |   1   |   –    |

> Ejemplo: "gato–alfombra" aparece 1 vez (oración 1), "perro–alfombra" 1 vez (oración 2), "ratón–cocina" 1 vez (oración 4), etc.

**2. Ajuste de vectores según razones de coocurrencia**

GloVe busca dos vectores por palabra (una "de entrada" y otra "de salida") de modo que, **sin ecuaciones**, la distancia o diferencia entre esos vectores refleje **la razón** entre sus frecuencias conjuntas.  

- Si dos palabras coocurren **mucho** juntas (como "ratón" y "queso"), sus vectores se acercan en el espacio.  
- Si coocurren **poco** (por ejemplo "gato" y "queso"), sus vectores quedan más separados.  
- Además, GloVe penaliza menos las coocurrencias **muy raras** o **muy frecuentes**, equilibrando la influencia de ambas.

**3. Ventaja práctica**

- **Información local**: aprovecha los pares de palabras que aparecen en cada ventana (contexto inmediato), igual que Skip‑gram o CBOW.  
- **Información global**: incorpora el conteo total de coocurrencias de todo el corpus, estabilizando la calidad del espacio vectorial cuando el vocabulario crece o hay palabras muy frecuentes.  

Así, los vectores resultantes combinan "lo que pasa alrededor" de cada palabra en frases individuales con "cómo suele comportarse" esa palabra a lo largo de **todo** el corpus, logrando una representación robusta y consistente.

#### Extensión morfológica con FastText 

FastText innova al descomponer cada palabra en subcomponentes, normalmente en forma de n‑gramas de caracteres. De esta manera, el embedding final es la suma o combinación de vectores de subunidades, lo que proporciona dos beneficios cruciales: capturar rasgos morfológicos y generar representaciones razonables para palabras no vistas durante el entrenamiento, al componerlas a partir de sus fragmentos. 

#### Uso en entornos industriales y académicos 

La disponibilidad de vectores preentrenados en grandes corpus, por ejemplo noticias, Wikipedia o Common Crawl, permite incorporar conocimiento 
semántico y sintáctico profundo sin entrenar desde cero. 

Herramientas como Gensim o la biblioteca oficial de FastText hacen trivial la carga y consulta de estos modelos, acelerando proyectos de clasificación de texto, análisis de sentimiento o sistemas de recomendación. 

### Modelos de representaciones contextualizadas 

#### ELMo: representaciones dependientes de la posición 

ELMo marcó un antes y un después al generar embeddings dinámicos que varían según la posición y el contexto completo de la frase. Internamente usa capas recurrentes bidireccionales que procesan el texto en ambas direcciones, permitiendo que cada vector combine información procedente tanto del pasado como del futuro inmediato. 

#### BERT: bidireccionalidad y preentrenamiento en dos fases 

BERT llevó la bidireccionalidad aún más lejos gracias al preentrenamiento con dos objetivos complementarios: ocultar aleatoriamente algunas palabras y predecirlas, y aprender a identificar si dos fragmentos de texto son contiguos en el texto original. Con esta receta, se crían poderosos encoders basados en arquitecturas Transformer que capturan interdependencias complejas y se afinan con una sola capa adicional para tareas específicas. 

#### GPT y variantes: generación y ajuste fino 

Por su parte, la familia GPT desarrolla representaciones autoregresivas, donde cada nueva palabra se predice exclusivamente a partir de lo previo. Al entrenar en enormes cantidades de texto, estos modelos dominan la generación coherente y la continuación de frases. Sobre ellos se apoya la técnica de **ajuste fino** (fine‑tuning) en tareas que requieren comprender preguntas y respuestas, generación creativa o diálogo conversacional, añadiendo capas ligeras que adaptan el modelo a dominios particulares. 

#### Ejemplos

**ELMo: representaciones dependientes de la posición**
  
Imagina la frase:  
> "Después de **leer** el libro, María **escribió** una reseña."

- Para la palabra "reseña", un embedding estático la representaría igual en cualquier oración.  
- **Con ELMo**, la representación de "reseña" en esta frase incorpora dos fuentes de información:  
  1. **Contexto hacia atrás**: que "escribió" y "María" preceden a "reseña", subrayando una acción creativa.  
  2. **Contexto hacia adelante**: que no hay palabras posteriores en este fragmento, pero la capa bidireccional aprende que va al final de la frase.  

**Detalles de la dinámica**  
- Cada palabra se convierte en **varios vectores**, uno por cada capa de la red recurrente bidireccional.  
- Para "reseña" se combinan estos vectores, de modo que el embedding resultante refleja si aparece tras un verbo de creación (como "escribió") o tras un sustantivo de objeto (por ejemplo, en "la reseña fue positiva").  
- Esto permite, por ejemplo, que la misma palabra "reseña" tenga un vector distinto si la frase fuese "La **reseña** generó controversia" (donde su función semántica es distinta).

**BERT: bidireccionalidad y preentrenamiento en dos fases**  
Frase de ejemplo:  

> "El clima en la costa es cálido, pero en la **montaña** hace frío."

1. **Objetivo de "máscara"**  
   - Se oculta aleatoriamente "montaña2: "El clima en la costa es cálido, pero en la `[MASK]` hace frío."  
   - El modelo debe adivinar "montaña" basándose en ambas mitades de la oración (antes y después de la máscara).

2. **Objetivo de "segmentos contiguos"**  
   - Se eligen dos fragmentos:  
     - Fragmento A: "El clima en la costa es cálido,"  
     - Fragmento B (aleatorio o adyacente): puede ser "pero en la montaña hace frío" (positivo) o "La economía global sube" (negativo).  
   - El modelo aprende a distinguir si B sigue realmente a A en el texto original.

**Ventaja práctica**  
- Al entrenar con estos dos objetivos, BERT capta relaciones **a gran escala** (cómo segmentar frases) y **a pequeña escala** (qué palabra falta), obteniendo un encoder que entiende interdependencias complejas a ambos niveles.  
- Posteriormente, para una tarea específica (clasificación de sentimientos, respuestas a preguntas, etiquetado de entidades), se añade una capa ligera que adapta ese conocimiento bidireccional sin tener que volver a entrenar todo el modelo desde cero.

**GPT y variantes: generación y ajuste fino**

Supón que entrenamos un modelo GPT con cientos de miles de novelas y artículos:

1. **Aprendizaje autoregresivo**  
   - Para completar "El viajero abrió la puerta y se encontró con un..." → el modelo predice "paisaje2, "descubrimiento", "misterio" u otras continuaciones, basándose únicamente en lo ya leído.  
   - Cada palabra nueva se genera condicionada por todas las anteriores, garantizando coherencia de estilo y tema.

2. **Ajuste fino en tareas específicas**  
   - **Diálogo conversacional**: partiendo del GPT base, se expone a ejemplos de preguntas y respuestas de soporte técnico.  
     - El modelo aprende a adoptar un tono amable y preciso.  
   - **Generación creativa**: se entrena con poemas y relatos cortos de distintos autores.  
     - El resultado es un GPT que, al recibir un inicio poético, continúa en un estilo que imita la métrica y el lenguaje aprendido.  
   - **Comprensión de preguntas**: se afinan capas finales con pares pregunta–respuesta de exámenes, logrando que el modelo seleccione la mejor respuesta de un conjunto dado.

**Ventaja práctica**  
- La arquitectura autoregresiva convierte a GPT en un generador de texto fluido, pues cada paso maximiza la probabilidad de la siguiente palabra.  
- El **fine‑tuning** permite especializar al modelo de forma eficiente: basta con un conjunto moderado de ejemplos etiquetados para adaptarlo a dominios muy concretos (legal, médico, creativo, etc.), sin sacrificar la capacidad de generación general aprendida en el preentrenamiento.

