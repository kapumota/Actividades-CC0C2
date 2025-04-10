
### Introducción a la evaluación en NLP
En los últimos años, el campo del procesamiento del lenguaje natural (NLP) ha experimentado avances notables, impulsados en gran parte por la adopción de técnicas de aprendizaje profundo. Esta evolución ha permitido el desarrollo de modelos capaces de realizar tareas tan diversas como la traducción automática, la generación de resúmenes, la respuesta a preguntas y la creación de chatbots que sostienen conversaciones casi humanas. Sin embargo, a medida que la complejidad y capacidad de estos modelos aumenta, también lo hace el reto de evaluar la calidad del texto que generan. Es aquí donde entran en juego diversas métricas de evaluación, diseñadas para cuantificar aspectos fundamentales de la producción textual y  medir, de forma objetiva, la calidad de las salidas generadas por estos sistemas.

El objetivo primordial de las métricas de evaluación es brindar una herramienta que permita a los investigadores y desarrolladores identificar las fortalezas y debilidades de un modelo  de NLP, facilitando la mejora y optimización de estos sistemas. En este informe se analizan en detalle cuatro métricas esenciales:  **Perplejidad**, **ROUGE**, **BLEU** y **METEOR**. 

Cada una de estas métricas aborda la evaluación desde una perspectiva diferente, permitiendo así un análisis multidimensional de la calidad del texto generado.

#### Perplejidad (perplexity)

La perplejidad es una métrica clásica y ampliamente utilizada en la evaluación de modelos de lenguaje. Esta métrica se basa en la capacidad del modelo para predecir la siguiente palabra en una secuencia. La idea central es cuantificar la incertidumbre del modelo al predecir el próximo elemento en una cadena de palabras. Matemáticamente, la perplejidad se derivadel concepto de entropía, la cual mide la incertidumbre o la sorpresa asociada a una distribución de probabilidad.

Cuando se evalúa un modelo, una perplejidad baja indica que el sistema tiene una mayor comprensión de las estructuras y patrones subyacentes en el lenguaje, permitiéndole predecir con mayor precisión cuál será la palabra siguiente. Por el contrario, una perplejidad alta sugiere que el modelo enfrenta dificultades para anticipar las secuencias lingüísticas, lo que puede deberse a una comprensión superficial del contexto o a limitaciones en el entrenamiento.

La aplicación de la perplejidad es particularmente relevante en tareas de generación de texto y modelado de lenguaje, donde la fluidez y coherencia son esenciales. En la práctica, se utiliza para comparar modelos, siendo uno de los indicadores clave en la determinación del rendimiento de un sistema. Es importante señalar que, aunque una perplejidad baja es deseable, esta métrica no captura todos los matices de la calidad textual, por lo que se complementa con otras medidas.

#### ROUGE: Medición de la calidad de resúmenes y textos generados

ROUGE, acrónimo de Recall-Oriented Understudy for Gisting Evaluation, es un conjunto de métricas utilizadas principalmente para evaluar sistemas de resumen automático y la calidad del  texto generado. A diferencia de la perplejidad, ROUGE se enfoca en la comparación directa entre un texto generado por el modelo (denominado "hipótesis” o H) y uno o varios  textos de referencia elaborados por humanos (denominados "referencia" o R). 
Esta comparación se realiza a través de la identificación y conteo de n-gramas coincidentes, así como mediante otras técnicas más sofisticadas.

#### ROUGE-N

ROUGE-N se centra en el recuento de n-gramas compartidos entre el texto generado y el de referencia. Los n-gramas son secuencias contiguas de "n" palabras, y la métrica puede  aplicarse a diferentes órdenes, como bigramas (n = 2), trigramas (n = 3), entre otros. La evaluación se realiza considerando dos aspectos fundamentales:  

- **Precisión:** Se calcula como la relación entre el número de n-gramas coincidentes en la hipótesis y el total de n-gramas presentes en la hipótesis.
  Este valor indica qué tan exacta es la generación del modelo en cuanto a la inclusión de n-gramas correctos.  
- **Recall:** Se determina dividiendo el número de n-gramas comunes entre la hipótesis y la referencia entre el total de n-gramas en la referencia.
  Este valor refleja la capacidad del modelo para cubrir la totalidad de información importante contenida en el texto de referencia.

Una vez obtencidos los valores de precisión y recall, se suele calcular la puntuación F1, que es la media armónica entre ambos.  La puntuación F1 permite obtener una medida única que resume el desempeño del modelo en términos de exactitud y exhaustividad.

#### ROUGE-L y la secuencia común más larga (LCS)

La métrica ROUGE-L introduce el concepto de la Secuencia Común Más Larga (LCS, por sus siglas en inglés), la cual mide la similitud entre el texto generado y la referencia considerando la secuencia más larga de palabras que aparece en ambos textos, sin que las palabras tengan que ser necesariamente consecutivas. 

El proceso de evaluación con ROUGE-L incluye:  

- **Identificación de la LCS:** Se determina la secuencia de palabras que se mantiene en el mismo orden en ambos textos, lo que permite capturar aspectos estructurales de la oración sin requerir que las palabras estén juntas.  
- **Cálculo de precisión y recall:** La precisión se obtiene dividiendo la longitud de la LCS por el número total de palabras en la hipótesis, mientras que el recall se calcula dividiendo la longitud de la LCS por el total de palabras en el texto de referencia.

Esta métrica es especialmente útil en escenarios donde el orden y la continuidad del contenido son fundamentales, ya que proporciona una visión más integral de la calidad textual  al tener en cuenta tanto la coincidencia de palabras como la estructura subyacente.

#### ROUGE-S y el emparejamiento con skip-gram

ROUGE-S amplía las capacidades de ROUGE-N y ROUGE-L mediante la consideración de skip-grams, que son pares de palabras que aparecen en la referencia y en la hipótesis, aunque no  necesariamente de forma consecutiva. Esta técnica permite reconocer coincidencias parciales en el texto generado, lo que resulta útil cuando el modelo introduce palabras adicionales  o varía ligeramente el orden de las palabras.  

- **Identificación de skip-grams:** Se identifican pares de palabras en la referencia y se verifica si estos aparecen en el texto generado, permitiendo saltarse palabras intermedias.  
- **Cálculo de la precisión y recall:** Al igual que en ROUGE-N, se calcula la precisión dividiendo el número de skip-grams comunes entre el total de skip-grams en la hipótesis, y el  recall dividiendo el número de skip-grams comunes entre el total de skip-grams en la referencia.

El uso de ROUGE-S es especialmente relevante en la evaluación de resúmenes, ya que permite capturar relaciones semánticas que pueden perderse al evaluar únicamente la coincidencia  exacta de n-gramas.

### BLEU: Evaluación de la calidad en traducción automática

La métrica BLEU (Bilingual Evaluation Understudy) ha sido ampliamente adoptada para evaluar la calidad de las traducciones automáticas. 
Su principal fortaleza radica en la capacidad de comparar la traducción generada por el modelo con una o varias traducciones de referencia, permitiendo evaluar la similitud entre ellas de manera cuantitativa. 

El funcionamiento de BLEU se basa en el recuento de n-gramas y la aplicación de una penalización por brevedad, garantizando así que la longitud de la traducción generada no distorsione la evaluación.

#### Cálculo de la precisión cortada (clipped precision)

Una de las características distintivas de BLEU es el uso de la precisión cortada. A diferencia de la precisión tradicional, la precisión cortada establece un límite superior en el  conteo de n-gramas coincidentes para evitar que las repeticiones excesivas en la traducción generada inflen artificialmente la puntuación. Por ejemplo, en un caso donde la hipótesis  contenga repeticiones innecesarias de ciertos términos, la precisión cortada limita la cantidad de coincidencias permitidas en función del máximo número de apariciones de ese n-grama en cualquier traducción de referencia.

Este proceso se realiza para cada orden de n-grama (por ejemplo, unigramas, bigramas, trigramas, etc.), y luego se combinan los valores utilizando una media geométrica ponderada. 
La fórmula implica calcular, para cada orden de n-grama, la razón entre el número de ocurrencias en la hipótesis que también aparecen en la 
referencia (limitada al máximo de ocurrencias permitidas) y el total de n-gramas en la hipótesis.

#### Penalización por brevedad

Además del cálculo de la precisión cortada, BLEU incorpora una penalización por brevedad que actúa cuando la traducción generada es significativamente más corta que las referencias. 
La idea es evitar que un modelo obtenga una puntuación alta simplemente por generar frases muy breves y altamente precisas en términos de coincidencia, pero que omitan información relevante. La penalización por brevedad se aplica multiplicativamente a la media geométrica de las precisiones, reduciendo la puntuación final cuando se detecta una discrepancia importante en la longitud.

#### Ejemplo de cálculo de BLEU

Para ilustrar el funcionamiento de BLEU, se puede imaginar un escenario en el que se evalúan unigramas, bigramas y trigramas. Se procede a contar los n-gramas coincidentes entre la traducción generada y las traducciones de referencia. Posteriormente, se calculan las precisiones para cada orden de n-grama y se combinan mediante una media geométrica. 
La aplicación final de la penalización por brevedad resulta en el cálculo definitivo del puntaje BLEU, el cual sirve como indicador de la calidad de la traducción generada.

La fortaleza de BLEU reside en su capacidad para evaluar traducciones cuando existen múltiples versiones válidas, lo que permite incluir diversas traducciones de referencia y, de esta forma, capturar la riqueza y variedad del lenguaje.


### METEOR: Evaluación con énfasis en el orden y la correspondencia semántica

La métrica METEOR (Metric for Evaluation of Translation with Explicit ORdering) representa otro enfoque para la evaluación de la calidad en traducción automática. 
A diferencia de BLEU, METEOR toma en cuenta no solo la coincidencia de palabras, sino también otros aspectos relevantes como la correspondencia semántica, la posibilidad de utilizar  sinónimos y la importancia del orden de las palabras. Esta métrica ha demostrado una alta correlación con las evaluaciones humanas, especialmente en tareas de traducción y generación de texto.

#### Proceso de alineación y conteo de unigramas

El primer paso en la evaluación con METEOR consiste en establecer una alineación entre las palabras del sistema de traducción (la hipótesis, H) y las de la traducción de referencia (R).
Esta alineación permite identificar las palabras coincidentes, denominadas unigramas, que se consideran un indicador directo de la similitud entre ambos textos. Una vez establecida la correspondencia, se cuenta el número total de coincidencias (m), que será fundamental para el cálculo de precisión y recall.

#### Cálculo de precisión y recall en METEOR

Siguiendo la alineación establecida, se procede a calcular la precisión como el cociente entre el número de coincidencias (m) y el total de palabras presentes en la traducción generada (H). De manera similar, el recall se obtiene dividiendo el mismo número de coincidencias (m) por el total de palabras de la traducción de referencia (R). 
Estas dos medidas permiten evaluar, respectivamente, la exactitud de la traducción en términos de inclusión de palabras correctas y la exhaustividad con la que se han capturado las palabras relevantes del texto de referencia.

#### Media armónica y balance entre precisión y recall

Para lograr un balance entre la precisión y el recall, METEOR utiliza una media armónica, lo cual se traduce en la obtención de un valor que refleja de forma equilibrada ambos aspectos. 
Este promedio parametrizado, que se obtiene calculando la inversa de la media aritmética de las inversas de la precisión y el recall, permite ponderar de manera justa las diferencias en ambas métricas. En el ejemplo planteado en el texto base, se asume un valor de α igual a 0.5 para el cálculo de la media armónica, lo que  resulta en un valor aproximado de 0.61.

#### Penalización por orden de palabras

Uno de los aspectos que distingue a METEOR es la inclusión de una penalización que mide la discrepancia en el orden de las palabras entre la traducción generada y la de referencia. 
Una vez que se han identificado los unigramas coincidentes, se determina cuántos segmentos consecutivos se pueden formar en la alineación. Cada segmento representa una secuencia de palabras que se encuentra en el mismo orden en ambos textos. Si la traducción generada rompe esta continuidad, se aplicará una penalización que depende del número de trozos en relación con el total de coincidencias. 

La penalización se modula mediante un parámetro γ, el cual, en el ejemplo mencionado, se asume con un valor de 0.8. Esta penalización asegura que la evaluación no solo valore la presencia de palabras correctas, sino también la coherencia en la estructura sintáctica y semántica del texto.

#### Cálculo final del puntaje METEOR

Una vez definidos todos los componentes anteriores, el puntaje METEOR se calcula combinando la media armónica de precisión y recall con la penalización por orden de palabras. 
Este valor final refleja la similitud global entre la traducción del sistema y la referencia, integrando tanto la correspondencia léxica como la estructural. 
La fortaleza de METEOR radica en su capacidad para reconocer coincidencias parciales y el uso de sinónimos, lo que lo hace particularmente adecuado en contextos donde existen múltiples formas válidas de expresar una misma idea.

### Integración y aplicación de las métricas en el desarrollo de modelos de NLP

El uso conjunto de estas métricas: Perplejidad, ROUGE, BLEU y METEOR, ofrece una perspectiva completa sobre el rendimiento de los modelos de NLP. Cada métrica aporta información valiosa que permite evaluar distintos aspectos del texto generado:

- **Perplejidad:** Brinda una medida directa de la capacidad del modelo para predecir secuencias de palabras y, en consecuencia, su comprensión del lenguaje subyacente. Es una métrica que se utiliza con frecuencia durante el entrenamiento del modelo para ajustar parámetros y mejorar la generación de texto.
  
- **ROUGE:** Con sus variantes (ROUGE-N, ROUGE-L y ROUGE-S), permite evaluar tanto la coincidencia exacta de n-gramas como la estructura general y la fluidez del contenido generado. Es especialmente útil en tareas de resumen automático, ya que compara de forma directa la salida del modelo con resúmenes de referencia elaborados por humanos.

- **BLEU:** Su aplicación se orienta primordialmente a la traducción automática, donde la existencia de múltiples traducciones válidas demanda una evaluación que contemple tanto la precisión en la correspondencia de n-gramas como la longitud adecuada de la salida. La combinación de precisión cortada y penalización por brevedad garantiza que el puntaje refleje de forma equilibrada la calidad de la traducción.

- **METEOR:** Complementa a BLEU al introducir consideraciones adicionales como la alineación de palabras, el uso de sinónimos y la evaluación del orden sintáctico. Esto lo convierte en una herramienta muy valiosa en escenarios en los que se requiere una evaluación más matizada de la similitud semántica entre la traducción generada y las traducciones de referencia.

En la práctica, estos métodos se utilizan en conjunto para obtener una visión holística del rendimiento de un modelo. Por ejemplo, durante el proceso de desarrollo y ajuste de un sistema de traducción automática, se pueden emplear tanto BLEU como METEOR para identificar casos en los que la traducción es precisa en términos de vocabulario, pero falla en capturar la estructura o el orden correcto de las palabras. Del mismo modo, en aplicaciones de resumen automático, el uso de ROUGE en sus distintas variantes permite detectar tanto la precisión en la selección de frases clave como la cohesión del resumen final.

La combinación de estas métricas resulta especialmente relevante cuando se trata de comparar modelos. Al aplicar estas evaluaciones a diferentes arquitecturas o técnicas de entrenamiento, los investigadores pueden identificar cuál de ellas ofrece mejores resultados en función de criterios cuantificables. Esto facilita la toma de decisiones informadas  sobre qué aspectos del modelo deben ser mejorados, permitiendo ajustar tanto la arquitectura interna como los datos de entrenamiento para optimizar el rendimiento general del sistema.

Otra ventaja importante es la capacidad de estas métricas para adaptarse a diferentes tareas. Aunque cada métrica se ha diseñado con un objetivo específico, su aplicación no se limita a una única tarea de NLP. Por ejemplo, mientras que BLEU y METEOR se han orientado históricamente a la traducción automática, sus principios subyacentes también pueden aplicarse a otros campos, como la generación de respuestas en chatbots o la síntesis de texto en sistemas de asistencia virtual. 

De igual manera, ROUGE es ampliamente empleado en la evaluación de resúmenes, pero sus variantes permiten evaluar cualquier tarea en la que la comparación de secuencias textuales  resulte relevante.

La integración de estas métricas en el ciclo de desarrollo de sistemas de NLP ha permitido que los modelos evolucionen hacia soluciones cada vez más sofisticadas y capaces de  manejar la complejidad del lenguaje humano. La evaluación cuantitativa, basada en parámetros bien definidos, contribuye a la identificación de patrones de error y a la implementación de estrategias de mejora continua. Además, la capacidad de estos indicadores para correlacionarse con la evaluación humana refuerza su utilidad como herramientas de benchmarking, facilitando la comparación entre diferentes enfoques y algoritmos.

Asimismo, la evolución de estas métricas ha ido de la mano con el progreso tecnológico en el ámbito del NLP. Con el surgimiento de modelos generativos cada vez más potentes, ha sido necesario desarrollar métricas que puedan capturar no solo la precisión léxica, sino también la coherencia, la fluidez y la capacidad de mantener contextos complejos a lo  largo de textos extensos. En este sentido, la investigación continua en el área busca perfeccionar estos métodos y, a su vez, desarrollar nuevos enfoques que permitan evaluar de forma aún más exhaustiva la calidad de los textos generados por inteligencia artificial.

Cada una de las métricas discutidas presenta ventajas y limitaciones que deben ser consideradas en función del contexto de aplicación. 
Mientras que la perplejidad ofrece una visión general de la capacidad predictiva del modelo, ROUGE, BLEU y METEOR permiten evaluar la calidad del contenido en relación con textos  de referencia, cada una aportando una perspectiva única en términos de precisión, estructura y correspondencia semántica. 


