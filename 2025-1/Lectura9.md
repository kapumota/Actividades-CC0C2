### LLM (Large Language Models)

Los grandes modelos de lenguaje (Large Language Models, o LLMs) han revolucionado el campo del procesamiento del lenguaje natural 
al combinar arquitecturas profundas, cantidades masivas de datos textuales y técnicas de entrenamiento avanzadas para aprender  las sutilezas del lenguaje humano. 

En esencia, un LLM es un modelo estadístico que, a partir de un gigantesco corpus de texto, que puede incluir libros, artículos,  páginas web y diálogos aprende a predecir la probabilidad de aparición de cada palabra o subunidad del lenguaje 
dada una secuencia de contextos anteriores. 

Esta capacidad predictiva es la base de su sorprendente destreza para tareas tan diversas como completar oraciones, traducir idiomas, responder preguntas, generar resúmenes o incluso componer poemas.

El núcleo arquitectónico de casi todos los LLMs modernos es el **Transformer**. 
A diferencia de las redes recurrentes o las convolucionales, un Transformer no procesa la secuencia de tokens de forma estrictamenteordenada, sino que aplica mecanismos de atención para ponderar la relevancia de cada token de entrada con respecto a los demás.

Gracias a la atención escalada y a las cabeceras múltiples, el Transformer puede capturar dependencias a muy largo plazo en el 
texto, por ejemplo la relación entre un sujeto mencionado al inicio de un párrafo y un pronombre situado cientos de  palabras después sin sufrir los problemas de desvanecimiento o explosión de gradientes propios de las arquitecturas recurrentes.

Para que un LLM trabaje con texto, primero debe convertir las palabras y subpalabras en representaciones numéricas. 
Este paso es la tokenización, un proceso que segmenta el texto en unidades discretas: tokens. Los enfoques más comunes incluyen  **Byte-Pair Encoding (BPE)**, **WordPiece** y **SentencePiece**. 

Bajo estos métodos, palabras muy frecuentes permanecen como tokens individuales, mientras que palabras raras se descomponen en subunidades más pequeñas, lo que ayuda al modelo a tratar eficientemente un vocabulario extenso y a generalizar mejor sobre términos desconocidos.

Una vez definidos los tokens, cada uno se asocia con un vector de embedding, un arreglo de decenas o cientos de dimensiones que el modelo aprende durante el entrenamiento. 
A este vector de embedding se le suma un vector de codificación posicional para incorporar información sobre la posición del token en la secuencia, de modo que, al no depender de una estructura
recurrente, el modelo pueda aún así reconocer el orden de las palabras. 

Estos embeddings enriquecidos fluyen a través de decenas o cientos de capas Transformer durante el preentrenamiento.

El preentrenamiento de un LLM suele basarse en uno de dos objetivos principales: el **modelado de lenguaje autoregresivo**, donde el modelo aprende a predecir el siguiente token en una secuencia basándose únicamente en los tokens previos, o el modelado de
lenguaje enmascarado (Masked Language Modeling), en el que algunos tokens de la secuencia original se ocultan y el modelo debe deducirlos. 
En ambos casos, el modelo ajusta sus cientos de millones o incluso miles de millones  de parámetros mediante métodos de optimización como Adam, alimentándose de miles de millones de tokens de texto.

Una vez preentrenado, un LLM puede utilizarse directamente para generar texto: dado un prompt o texto inicial, el modelo predice el token siguiente, lo incorpora al prompt y repite el proceso hasta completar la respuesta. A este proceso de generación se le llama inference o predicción. Según la estrategia de muestreo que se elija, la salida puede tomar formas muy distintas: una búsqueda voraz (greedy) siempre selecciona el token de mayor probabilidad, mientras que métodos más sofisticados —como beam search, sampling con temperatura o top-k/top-p sampling— introducen aleatoriedad o mantienen varias hipótesis simultáneas para equilibrar coherencia y creatividad.

El tamaño de contexto de un LLM es otra variable fundamental: define cuántos tokens anteriores el modelo puede tener en cuenta al generar cada nuevo token. 
Modelos tempranos como GPT-2 manejaban hasta 1 024 o 2 048 tokens, mientras que arquitecturas más recientes alcanzan 32 k o incluso 100 k tokens de contexto, lo cual permite procesar documentos enteros o conversaciones extensas sin necesidad de fragmentarlos.

A medida que crecieron los modelos, investigadores descubrieron que su comportamiento seguía patrones previsibles, las llamadas **leyes de escalamiento**. 
Estas permiten estimar cuánto mejora la calidad del modelo, medida en pérdida de validación o en precisión sobre tareas estándares. Si, por ejemplo, duplicamos su número de parámetros o duplicamos la cantidad de datos de entrenamiento. 
Gracias a estas leyes, los laboratorios pueden planificar los recursos computacionales necesarios para alcanzar un determinado nivel de desempeño.

Más sorprendente aún ha sido el fenómeno de las **habilidades emergentes**. A partir de cierto umbral de tamaño y cantidad de datos, los modelos comienzan a exhibir capacidades que no tenían los modelos más pequeños: razonamiento lógico, analogías complejas, comprensión de instrucciones detalladas y hasta cierta forma de sentido común rudimentario. 

Estas emergencias sugieren que, al aumentar la escala, el modelo desarrolla internamente representaciones más abstractas del lenguaje y del mundo.

Sin embargo, los LLMs no son infalibles. A pesar de su potencia, tienden a alucinar es decir, a generar afirmaciones factualmente incorrectas que suenan plausibles y a reproducir **sesgos** presentes en sus datos de entrenamiento, tanto de género como raciales, ideológicos o culturales. 
Mitigar estos problemas es un área activa de investigación: se emplean filtros adicionales, entrenamiento adversarial, calibración de salidas y verificación con fuentes externas.

Una de las técnicas más efectivas para controlar la salida de un LLM sin reentrenarlo es el uso de **prompts** cuidadosamente diseñados. 
Al proporcionar ejemplos de entrada y salida en el propio prompt (**few-shot learning**), el modelo puede adaptarse a formatos específicos, estilos de lenguaje o requerimientos de la tarea. 
Incluso sin ejemplos, un prompt que describa claramente la instrucción y el contexto puede mejorar drásticamente los resultados.

En el panorama actual de LLMs, conviven diversas familias y proveedores. Por un lado, modelos como **GPT** (de OpenAI), **PaLM** (de Google) y **LLaMA** (de Meta) dominan el mercado comercial y de investigación, mientras que iniciativas de código abierto como Bloom (del proyecto BigScience) democratizan el acceso. Cada opción difiere en licencia, número de parámetros, tamaño de vocabulario, 
estrategias de entrenamiento y coste de inferencia.

En la práctica, los LLMs han saltado rápidamente de la investigación al uso cotidiano: chatbots conversacionales, asistentes de programación, generación de contenidos creativos, sistemas de resumen, herramientas de análisis de sentimiento, traducción automática y motores de búsqueda extremadamente sofisticados. 
Para cada caso, la elección del modelo depende de múltiples factores: latencia aceptable, coste por token, privacidad de los datos y longitud de contexto necesaria.

Para facilitar el desarrollo de aplicaciones con LLMs surge la **ingeniería de prompting**, una disciplina que estudia cómo estructurar y refinar los prompts para obtener respuestas óptimas. Más allá de redactar la instrucción, el prompting puede incorporar estrategias como el chain-of-thought, que pide al modelo razonar en pasos intermedios, o el self-ask, en el que el modelo se
formula subpreguntas antes de responder a la consulta principal.

En paralelo, han aparecido frameworks especializados que abstraen la interacción con los LLMs y facilitan la construcción de pipelines más complejos. 
**LangChain**, por ejemplo, permite encadenar múltiples pasos de generación y procesamiento: se define un PromptTemplate con variables, se conecta a un proveedor de LLM (OpenAI, Hugging Face u otro) y se crea un **LLMChain** que procesa entradas, ejecuta lógica intermedia y genera salidas. 

Además, LangChain ofrece **"memory"** para manejar contextos conversacionales y **"agents"** que pueden invocar herramientas externas, como APIs, motores de búsqueda o bases de datos  bajo demanda del propio modelo.

Otra pieza esencial es **LlamaIndex** (antes GPT-Index), centrada en construir índices vectoriales de documentos. Cuando necesitamos que el LLM responda con datos fiables y actualizados, en lugar de confiar solo en el conocimiento implícito del modelo, utilizamos **Retrieval-Augmented Generation (RAG)**. 

Primero indexamos un corpus en vectores de embeddings y, ante una consulta, recuperamos los fragmentos más relevantes. Luego inyectamos esos fragmentos en el prompt para que el modelo los use como evidencia, reduciendo alucinaciones.

En implementaciones avanzadas de **RAG**, podemos combinar búsquedas bag-of-words (BM25) con recuperación por embeddings, aplicar modelos de reranking para ordenar pasajes, paginar el contexto cuando es muy extenso y hasta ejecutar cadenas de razonamiento interno sobre la información recuperada (chain-of-thought RAG). De este modo, conseguimos respuestas fundamentadas y razonadas sobre bases de información específicas, como documentos corporativos, bases de datos o la web en tiempo real.

Los agentes, a su vez, son instancias de LLMs que planifican y coordinan varias acciones: pueden decidir si necesitan llamar a un motor de búsqueda, acceder a una API climática o ejecutar un script interno, todo ello en base a la instrucción recibida. LangChain incluye un toolkit que define las herramientas disponibles y el formato de las llamadas, y el LLM aprende a ensamblar esos pasos de forma coherente.

Cuando la personalización va más allá de lo que permite el prompting, entran en juego las técnicas de **fine-tuning**. 

Además del ajuste fino completo  que puede ser muy costoso,  existen métodos más ligeros como **LoRA (Low-Rank Adaptation)**, donde solo entrenamos matrices de baja dimensión, o **QLoRA**, que aplica cuantización de baja precisión para reducir el uso de memoria durante el ajuste. Los adapters, por su parte, insertan pequeñas capas entrenables dentro de un modelo congelado, logrando 
adaptaciones de alta calidad con un coste muy inferior al reentrenamiento total.

Finalmente, el despliegue de LLMs implica decisiones cruciales de infraestructura. Muchas organizaciones optan por endpoints gestionados en la nube, pagando por token y olvidándose del mantenimiento del hardware. Otras requieren despliegues on-premise, instalando contenedores optimizados con TensorRT u ONNX Runtime y aprovechando GPUs locales o clusters de GPU. En todos los casos, es esencial garantizar escalado automático, balanceo de carga, caching de respuestas frecuentes y monitorización de latencia y coste.

En conjunto, los LLMs representan el avance actual del procesamiento del lenguaje natural: desde su fundamento en Transformers y modelado de lenguaje, pasando por tokenización y embeddings, hasta entrenamientos masivos, generación avanzada, mecanismos de recuperación y despliegues industriales. Aun cuando persisten desafíos como las alucinaciones y los sesgos, el desarrollo de técnicas de prompting, RAG, agentes y fine-tuning nos permite construir aplicaciones cada vez más confiables, eficientes y adaptadas a necesidades concretas. La combinación de estos avances señala un futuro en el que la interacción con sistemas inteligentes basados en texto será tan natural y versátil como conversar con un experto humano.
