## Lista de proyecto de curso

Estructura del repositorio: Cada estudiante debe crear un repositorio que contenga 3 carpetas principales, una para cada entrega gradual del proyecto.

Las carpetas deben nombrarse de la siguiente manera:

- Entrega_1 (para el primer entregable)
- Entrega_2 (para el segundo entregable)
- Entrega_3 (para el tercer y último entregable)

Contenido de cada carpeta: En cada una de estas carpetas, debe reflejarse el avance del proyecto correspondiente a la fecha de entrega. Los cambios realizados entre entregas deben documentarse claramente en los archivos de la carpeta.

Actualización de cambios graduales: Asegúrense de que cada entrega muestre los avances y mejoras hechas desde la entrega anterior. Utilicen el control de versiones (commits) para registrar cada cambio significativo.

Plazo de entrega:

Entrega 1: hasta el 30 de noviembre
Entrega 2: hasta el 7 de diciembre
Entrega 3 (final): hasta el 14 de diciembre

Documentación de cambios: Incluir en cada carpeta un archivo README.md que explique los cambios realizados en esa etapa y cómo estos avances reflejan los conceptos enseñados en clase.

---

### **Proyecto 1: Ajustar finamente un LLM con PPO vs DPO vs ORPO utilizando el paquete PEFT**

**Descripción detallada:**

El ajuste fino de modelos de lenguaje a gran escala (LLMs) es una tarea esencial para adaptar modelos preentrenados a aplicaciones específicas. En este proyecto, se propone comparar tres técnicas de optimización por políticas: Proximal Policy Optimization (PPO), Deterministic Policy Optimization (DPO) y Optimistic Regularized Policy Optimization (ORPO), utilizando el paquete PEFT (Parameter-Efficient Fine-Tuning). El objetivo es evaluar cuál de estos métodos ofrece el mejor rendimiento en términos de eficiencia y precisión, especialmente en entornos con recursos computacionales limitados.

**Objetivos específicos:**

1. **Comprender y aplicar las técnicas de PPO, DPO y ORPO:** Investigar la teoría detrás de cada método, entender sus ventajas y limitaciones, y aprender a implementarlos en el contexto del ajuste fino de LLMs.

2. **Utilizar PEFT para optimizar el ajuste fino:** Aprovechar las capacidades de PEFT para ajustar modelos grandes sin necesidad de modificar todos los parámetros, lo que reduce significativamente los requerimientos computacionales.

3. **Comparar el rendimiento de los modelos ajustados:** Evaluar cada modelo ajustado en tareas específicas, analizando métricas como precisión, tiempo de entrenamiento y eficiencia de recursos.

**Metodología:**

- **Preparación del entorno:**
  - Instalar y configurar el entorno de desarrollo, incluyendo bibliotecas como PyTorch, Transformers de Hugging Face y PEFT.
  - Verificar que el hardware disponible sea adecuado para el ajuste fino eficiente, aprovechando GPUs si están disponibles.

- **Selección del modelo base y conjunto de datos:**
  - Elegir un modelo de lenguaje preentrenado apropiado, como GPT-2 o BERT, según la tarea específica.
  - Seleccionar un conjunto de datos relevante para la tarea de ajuste fino, asegurando que sea de tamaño manejable pero suficientemente representativo.

- **Implementación de las técnicas de optimización:**
  - **PPO:** Implementar Proximal Policy Optimization, que es una técnica basada en gradientes que busca mejorar la estabilidad y eficiencia en el entrenamiento de modelos de políticas.
  - **DPO:** Aplicar Deterministic Policy Optimization, enfocándose en optimizar políticas determinísticas y analizando cómo afecta al rendimiento del modelo.
  - **ORPO:** Implementar Optimistic Regularized Policy Optimization, que introduce regularización optimista para mejorar la exploración durante el entrenamiento.

- **Ajuste fino con PEFT:**
  - Utilizar PEFT para ajustar sólo una pequeña fracción de los parámetros del modelo, manteniendo el resto congelado. Esto incluye técnicas como LoRA (Low-Rank Adaptation) y Adapters.

- **Entrenamiento y evaluación:**
  - Entrenar el modelo ajustado con cada técnica, monitoreando métricas como pérdida, precisión y tiempo de entrenamiento.
  - Evaluar los modelos en un conjunto de pruebas independiente para medir su rendimiento real.

- **Análisis comparativo:**
  - Comparar los resultados obtenidos con cada método, identificando fortalezas y debilidades.
  - Analizar el impacto de PEFT en la eficiencia y rendimiento del ajuste fino.

**Consideraciones técnicas:**

- **Requerimientos computacionales:** Aunque PEFT reduce la necesidad de recursos, es importante optimizar el código y utilizar técnicas como el procesamiento por lotes y la paralelización para mejorar la eficiencia.

- **Gestión de hiperparámetros:** Ajustar adecuadamente hiperparámetros como la tasa de aprendizaje, el tamaño de lote y los coeficientes de regularización es crucial para obtener buenos resultados.

- **Estabilidad del entrenamiento:** Algunas técnicas de optimización pueden ser sensibles a la inicialización y configuración, por lo que es necesario monitorear el entrenamiento y realizar ajustes según sea necesario.

**Posibles desafíos:**

- **Complejidad de implementación:** Entender y aplicar correctamente cada técnica de optimización puede ser complejo, requiriendo un estudio profundo de la literatura y prácticas recomendadas.

- **Limitaciones de hardware:** A pesar de utilizar PEFT, los modelos LLM siguen siendo grandes, y puede haber restricciones en el hardware disponible que afecten el tiempo de entrenamiento y la capacidad de experimentar con diferentes configuraciones.

- **Interpretación de resultados:** Analizar las diferencias en rendimiento entre las técnicas puede ser sutil, y es importante utilizar métricas apropiadas y realizar pruebas estadísticas cuando sea necesario.

**Impacto esperado:**

Al finalizar el proyecto, se espera tener una comprensión clara de cómo las diferentes técnicas de optimización por políticas afectan el ajuste fino de LLMs utilizando PEFT. Los resultados pueden contribuir a la comunidad proporcionando insights sobre cómo elegir el método más adecuado según los recursos disponibles y los requisitos de la tarea. Además, el proyecto demostrará la viabilidad de ajustar finamente modelos grandes en entornos con recursos limitados, lo que puede democratizar el acceso a tecnologías avanzadas de procesamiento del lenguaje natural.

---

### **Proyecto 2: Crear una aplicación interactiva de chat que utiliza GPT para responder en tiempo real, con soporte para WebSockets para comunicación continua**

**Descripción detallada:**

El objetivo de este proyecto es desarrollar una aplicación de chat en tiempo real que permita a los usuarios interactuar con un modelo GPT, recibiendo respuestas inmediatas y fluidas. La aplicación utilizará WebSockets para mantener una conexión bidireccional persistente entre el cliente y el servidor, facilitando una comunicación continua sin necesidad de realizar solicitudes HTTP repetidas.

**Objetivos específicos:**

1. **Desarrollar un backend robusto:** Implementar un servidor capaz de manejar múltiples conexiones simultáneas, integrar el modelo GPT y gestionar las interacciones con los usuarios de manera eficiente.

2. **Crear un frontend interactivo y amigable:** Diseñar una interfaz de usuario que sea intuitiva y responsiva, permitiendo a los usuarios comunicarse fácilmente con el modelo.

3. **Implementar comunicación en tiempo real con WebSockets:** Establecer una conexión estable y eficiente entre el cliente y el servidor para garantizar respuestas inmediatas y una experiencia de usuario fluida.

**Metodología:**

- **Configuración del entorno y herramientas:**
  - Seleccionar el stack tecnológico adecuado, como Node.js con Express para el backend y React para el frontend.
  - Utilizar bibliotecas como Socket.IO para facilitar la implementación de WebSockets.

- **Integración con GPT:**
  - Decidir si se utilizará una API externa (como OpenAI's GPT-3) o un modelo alojado localmente.
  - Implementar la lógica para enviar consultas al modelo y recibir respuestas, manejando casos de error y tiempos de espera.

- **Desarrollo del backend:**
  - Configurar el servidor para manejar conexiones WebSocket, autenticación de usuarios y gestión de sesiones.
  - Implementar middleware para procesar y filtrar entradas de usuario, previniendo inyecciones de código u otro contenido malicioso.

- **Diseño del frontend:**
  - Crear una interfaz de chat con elementos como historial de mensajes, indicación de escritura y notificaciones.
  - Asegurar la compatibilidad con dispositivos móviles y diferentes navegadores.

- **Implementación de WebSockets:**
  - Establecer la conexión en tiempo real entre el cliente y el servidor.
  - Manejar eventos como conexión, desconexión, recepción y envío de mensajes.

- **Optimización y pruebas:**
  - Realizar pruebas de carga para asegurar que el servidor puede manejar múltiples usuarios.
  - Optimizar el rendimiento minimizando la latencia y el uso de recursos.

**Consideraciones técnicas:**

- **Seguridad:** Implementar medidas para proteger la comunicación, como el uso de HTTPS y cifrado en WebSockets (WSS), además de validar y sanitizar todas las entradas de usuario.

- **Escalabilidad:** Diseñar el sistema pensando en la posibilidad de escalar horizontalmente si es necesario, utilizando balanceadores de carga y servicios en la nube.

- **Gestión de estado:** Decidir cómo se manejará el estado de la conversación, especialmente si se requiere mantener contexto entre mensajes.

**Posibles desafíos:**

- **Limitaciones de la API de GPT:** Si se utiliza una API externa, puede haber restricciones en la tasa de solicitudes o costos asociados que deben gestionarse adecuadamente.

- **Latencia y rendimiento:** Asegurar respuestas en tiempo real puede ser desafiante, especialmente si el modelo GPT tiene tiempos de procesamiento elevados.

- **Experiencia de usuario:** Crear una interfaz que sea intuitiva y atractiva requiere atención al detalle y posiblemente iteraciones basadas en feedback de usuarios.

**Impacto esperado:**

La aplicación resultante ofrecerá a los usuarios una experiencia interactiva con un modelo GPT, lo que puede tener aplicaciones en atención al cliente, educación, entretenimiento y más. Además, el proyecto servirá como base para futuras aplicaciones que requieran comunicación en tiempo real con modelos de lenguaje, demostrando cómo integrar tecnologías avanzadas de IA en aplicaciones web modernas.

---

### **Proyecto 3: Entrenar y ajustar un LLM especializado en la clasificación de noticias por temas, usando técnicas de fine-tuning y transfer learning**

**Descripción detallada:**

Este proyecto se enfoca en adaptar un modelo de lenguaje preentrenado para la tarea específica de clasificar noticias en distintas categorías temáticas, como política, economía, deportes, tecnología, entre otras. Utilizando técnicas de transfer learning y fine-tuning, el modelo aprovechará el conocimiento general adquirido durante el preentrenamiento y lo ajustará para desempeñarse óptimamente en la tarea de clasificación.

**Objetivos específicos:**

1. **Recolectar y preparar un conjunto de datos amplio y representativo:** Obtener noticias de diversas fuentes y asegurarse de que estén equilibradas entre las diferentes categorías.

2. **Seleccionar y adaptar un modelo preentrenado:** Elegir un modelo como DistilBERT, que es más ligero y adecuado para entrenar en una PC estándar.

3. **Entrenar y validar el modelo ajustado:** Realizar el fine-tuning del modelo y evaluar su rendimiento utilizando métricas como precisión, recall y F1-score.

**Metodología:**

- **Recolección y preprocesamiento de datos:**
  - Obtener noticias de fuentes confiables y etiquetarlas según su categoría temática.
  - Limpiar el texto eliminando caracteres especiales, etiquetas HTML y otros elementos no deseados.
  - Dividir el conjunto de datos en entrenamiento, validación y prueba.

- **Análisis exploratorio de datos:**
  - Evaluar la distribución de noticias entre las categorías para detectar desequilibrios.
  - Realizar visualizaciones para entender mejor las características de los datos.

- **Selección del modelo preentrenado:**
  - Optar por DistilBERT u otro modelo ligero que balancee rendimiento y eficiencia.
  - Cargar el modelo utilizando bibliotecas como Transformers de Hugging Face.

- **Ajuste fino del modelo:**
  - Configurar el modelo para la tarea de clasificación, agregando una capa de salida adecuada.
  - Entrenar el modelo con el conjunto de datos preparado, ajustando hiperparámetros como la tasa de aprendizaje y el número de épocas.
  - Utilizar técnicas como el early stopping para prevenir el sobreajuste.

- **Evaluación del modelo:**
  - Aplicar el modelo al conjunto de prueba y calcular métricas de rendimiento.
  - Analizar errores comunes y áreas de mejora.

- **Despliegue del modelo:**
  - Crear una API o interfaz que permita utilizar el modelo en aplicaciones reales.
  - Documentar el proceso y proporcionar instrucciones para la implementación.

**Consideraciones técnicas:**

- **Balance de clases:** Si hay categorías con pocas muestras, considerar técnicas como sobremuestreo o submuestreo para equilibrar el conjunto de datos.

- **Optimización de recursos:** Dado que se trabaja en una PC estándar, es importante manejar el tamaño de los lotes y utilizar técnicas como el gradiente acumulativo para entrenar eficientemente.

- **Actualización del modelo:** Las noticias son dinámicas y las tendencias cambian, por lo que es importante planificar cómo actualizar el modelo con nuevos datos.

**Posibles desafíos:**

- **Calidad y representatividad de los datos:** Asegurar que las noticias recopiladas sean de alta calidad y representen adecuadamente cada categoría.

- **Limitaciones computacionales:** Entrenar modelos incluso ligeros puede ser intensivo, por lo que es crucial optimizar el proceso y posiblemente utilizar servicios en la nube si es necesario.

- **Interpretabilidad del modelo:** En aplicaciones críticas, puede ser necesario entender por qué el modelo clasifica una noticia de cierta manera, lo que requiere técnicas de interpretabilidad.

**Impacto esperado:**

El modelo resultante permitirá clasificar automáticamente noticias, lo que puede ser útil para medios de comunicación, agregadores de noticias y herramientas de análisis de información. Además, el proyecto demostrará cómo adaptar eficientemente LLMs para tareas específicas, contribuyendo al campo del procesamiento del lenguaje natural y sirviendo como base para proyectos más complejos.

---

### **Proyecto 4: Usar embeddings contextuales para generar recomendaciones de películas basadas en descripciones de trama y preferencias del usuario**

**Descripción detallada:**

Este proyecto tiene como objetivo desarrollar un sistema de recomendación de películas que utilice embeddings contextuales para comprender tanto las descripciones de las películas como las preferencias de los usuarios. Al representar películas y preferencias en un espacio vectorial, el sistema puede calcular similitudes y sugerir películas que se alineen con los intereses del usuario.

**Objetivos específicos:**

1. **Generar embeddings de descripciones de películas:** Utilizar modelos preentrenados para obtener representaciones vectoriales de las descripciones.

2. **Capturar y procesar las preferencias del usuario:** Convertir las opiniones o intereses del usuario en embeddings compatibles.

3. **Implementar un algoritmo de recomendación basado en similitud de embeddings:** Sugerir películas que estén cerca en el espacio vectorial de las preferencias del usuario.

**Metodología:**

- **Recolección y preprocesamiento de datos:**
  - Obtener un conjunto de datos de películas que incluya descripciones, géneros, actores, etc.
  - Preprocesar las descripciones para eliminar ruido y normalizar el texto.

- **Generación de embeddings:**
  - Utilizar modelos como Sentence-BERT para obtener embeddings de las descripciones.
  - Almacenar los embeddings en una estructura de datos eficiente para búsquedas rápidas.

- **Captura de preferencias del usuario:**
  - Desarrollar un cuestionario o interfaz donde el usuario pueda expresar sus intereses.
  - Procesar las entradas del usuario y convertirlas en embeddings.

- **Cálculo de similitud y recomendación:**
  - Utilizar métricas como la similitud del coseno para calcular la cercanía entre embeddings.
  - Ordenar las películas según la similitud y presentar las mejores opciones al usuario.

- **Evaluación y mejora del sistema:**
  - Obtener feedback de los usuarios sobre las recomendaciones.
  - Ajustar el modelo y los parámetros según los resultados.

**Consideraciones técnicas:**

- **Eficiencia en búsquedas:** Para manejar grandes conjuntos de datos, puede ser necesario utilizar estructuras como árboles de búsqueda o técnicas de aproximación para acelerar las consultas.

- **Personalización avanzada:** Incorporar información adicional como el historial de visualización o calificaciones para mejorar las recomendaciones.

- **Diversidad en las recomendaciones:** Evitar presentar sólo opciones similares y ofrecer variedad para enriquecer la experiencia del usuario.

**Posibles desafíos:**

- **Representación adecuada de preferencias:** Los gustos de los usuarios pueden ser complejos, y capturarlos en un embedding puede requerir un enfoque sofisticado.

- **Limitaciones de los embeddings preentrenados:** Los modelos preentrenados pueden no capturar matices específicos del dominio de las películas, lo que podría afectar la calidad de las recomendaciones.

- **Actualización de datos:** El catálogo de películas es dinámico, por lo que es necesario actualizar regularmente los embeddings y la base de datos.

**Impacto esperado:**

El sistema proporcionará recomendaciones personalizadas de alta calidad, mejorando la experiencia del usuario al descubrir nuevas películas que se ajusten a sus intereses. Además, el proyecto demostrará cómo utilizar embeddings contextuales en sistemas de recomendación, lo que puede aplicarse a otros dominios como música, libros o productos.

---

### **Proyecto 5: Desplegar una API de preguntas y respuestas basada en LLM finamente ajustado con búsqueda de documentos**

**Descripción detallada:**

El objetivo de este proyecto es crear una API que pueda responder preguntas de los usuarios utilizando un modelo de lenguaje ajustado y una base de datos de documentos. El sistema combinará las capacidades del LLM para comprender y generar lenguaje natural con una búsqueda eficiente de información en los documentos para proporcionar respuestas precisas y contextualizadas.

**Objetivos específicos:**

1. **Ajustar finamente un LLM para tareas de preguntas y respuestas:** Adaptar el modelo para que pueda comprender preguntas y extraer información relevante.

2. **Implementar una base de datos de documentos con búsqueda eficiente:** Indexar los documentos de manera que se pueda acceder rápidamente a la información necesaria.

3. **Desarrollar y desplegar una API robusta:** Permitir que otros sistemas o aplicaciones puedan interactuar con el modelo a través de la API.

**Metodología:**

- **Preparación del LLM:**
  - Seleccionar un modelo preentrenado adecuado, como BERT o RoBERTa.
  - Ajustar el modelo utilizando un conjunto de datos de preguntas y respuestas, como SQuAD.
  - Evaluar el rendimiento del modelo en tareas de comprensión y generación de respuestas.

- **Implementación de la base de datos de documentos:**
  - Recolectar y organizar los documentos que serán la fuente de información.
  - Indexar los documentos utilizando motores de búsqueda como Elasticsearch o Apache Lucene.
  - Implementar técnicas de recuperación de información, como TF-IDF o modelos de lenguaje más avanzados.

- **Desarrollo de la lógica de respuesta:**
  - Cuando se recibe una pregunta, utilizar la búsqueda para obtener fragmentos relevantes de los documentos.
  - Pasar los fragmentos al LLM para generar una respuesta precisa y coherente.
  - Manejar casos en los que no se encuentra información relevante.

- **Desarrollo de la API:**
  - Diseñar endpoints claros y seguros para recibir preguntas y entregar respuestas.
  - Implementar autenticación y control de acceso si es necesario.
  - Documentar la API para facilitar su uso por parte de desarrolladores.

**Consideraciones técnicas:**

- **Escalabilidad y rendimiento:** Asegurar que la API pueda manejar múltiples solicitudes simultáneas y que las respuestas sean entregadas en un tiempo razonable.

- **Seguridad y privacidad:** Proteger los datos y las comunicaciones, especialmente si los documentos contienen información sensible.

- **Actualización de la base de datos:** Implementar mecanismos para agregar, eliminar o actualizar documentos sin interrumpir el servicio.

**Posibles desafíos:**

- **Calidad de las respuestas:** Garantizar que el modelo proporcione respuestas precisas y evitar generar información incorrecta o engañosa.

- **Gestión de ambigüedades:** Manejar preguntas ambiguas o mal formuladas, proporcionando aclaraciones o solicitando más información al usuario.

- **Limitaciones del LLM:** Los modelos tienen limitaciones y pueden requerir ajustes continuos para mejorar su rendimiento en tareas específicas.

**Impacto esperado:**

La API permitirá integrar capacidades avanzadas de preguntas y respuestas en diversas aplicaciones, como asistentes virtuales, sistemas de soporte al cliente o herramientas educativas. Al combinar la comprensión del lenguaje natural con una búsqueda eficiente de documentos, el sistema ofrecerá respuestas precisas y útiles, mejorando la interacción entre los usuarios y la información.

---

### **Proyecto 6: Utilizar Transformers para detectar anomalías en datos de series temporales, con aplicación en áreas como la monitorización de servidores o sistemas financieros**

**Descripción detallada:**

Este proyecto se centra en la implementación de modelos basados en Transformers para la detección de anomalías en series temporales. La detección de anomalías es crucial en diversos campos, como la monitorización de servidores, sistemas financieros, detección de fraudes, mantenimiento predictivo, entre otros. Los Transformers, originalmente diseñados para tareas de procesamiento del lenguaje natural, han demostrado ser efectivos en el modelado de dependencias a largo plazo, lo que los hace adecuados para analizar series temporales complejas.

**Objetivos específicos:**

1. **Desarrollar un modelo Transformer adaptado a series temporales:** Modificar y adaptar la arquitectura de Transformers para que sea efectiva en el análisis de datos secuenciales.

2. **Detectar anomalías con alta precisión:** Entrenar el modelo para identificar patrones normales y detectar desviaciones significativas que indiquen anomalías.

3. **Validar el modelo en datos reales:** Aplicar el modelo en conjuntos de datos reales de dominios como la monitorización de servidores o transacciones financieras.

**Metodología:**

- **Análisis del problema y definición de anomalías:**
  - Definir claramente qué se considera una anomalía en el contexto específico (p. ej., picos inusuales en el tráfico de red, transacciones financieras sospechosas).
  - Establecer criterios para la evaluación del modelo, como tasas de verdaderos positivos y falsos positivos.

- **Recolección y preprocesamiento de datos:**
  - Obtener conjuntos de datos históricos de series temporales relevantes, asegurando que incluyan ejemplos de anomalías conocidas.
  - Preprocesar los datos para manejar valores faltantes, normalizar escalas y resamplear si es necesario.

- **Adaptación del modelo Transformer:**
  - Modificar la arquitectura del Transformer para manejar datos numéricos secuenciales en lugar de texto.
  - Considerar el uso de modelos como el Time Series Transformer, que están específicamente diseñados para series temporales.

- **Entrenamiento del modelo:**
  - Entrenar el modelo en datos históricos, utilizando una parte de los datos para validación.
  - Implementar técnicas de entrenamiento supervisado o no supervisado, dependiendo de la disponibilidad de etiquetas de anomalías.

- **Detección de anomalías:**
  - Utilizar el modelo para predecir valores futuros y comparar con los valores reales, identificando desviaciones significativas.
  - Implementar umbrales o criterios estadísticos para determinar cuándo una desviación es considerada una anomalía.

- **Evaluación y ajuste del modelo:**
  - Evaluar el rendimiento del modelo utilizando métricas como precisión, recall, F1-score y área bajo la curva ROC.
  - Ajustar hiperparámetros y realizar validación cruzada para optimizar el rendimiento.

**Consideraciones técnicas:**

- **Eficiencia computacional:** Los Transformers pueden ser intensivos en cómputo, especialmente con secuencias largas. Es importante optimizar el modelo y considerar técnicas como la reducción de la dimensionalidad o la ventana deslizante.

- **Gestión de secuencias largas:** Implementar mecanismos para manejar dependencias a largo plazo sin sobrecargar la memoria, como el uso de Transformers eficientes o la segmentación de secuencias.

- **Interpretabilidad del modelo:** En aplicaciones críticas, es importante entender por qué el modelo detecta una anomalía. Se pueden utilizar técnicas de interpretabilidad para explicar las decisiones del modelo.

**Posibles desafíos:**

- **Desequilibrio en los datos:** Las anomalías suelen ser eventos raros, lo que puede causar desequilibrio en el conjunto de datos y afectar el entrenamiento del modelo. Es necesario manejar esto con técnicas como el sobremuestreo o la generación de datos sintéticos.

- **Detección de falsos positivos:** Un alto número de falsos positivos puede disminuir la confianza en el sistema. Es crucial ajustar los umbrales y criterios de detección para minimizar este problema.

- **Cambios en los patrones de datos:** Los patrones normales pueden cambiar con el tiempo, lo que requiere que el modelo se actualice o adapte para mantener su eficacia.

**Impacto esperado:**

La implementación exitosa de este proyecto proporcionará una herramienta poderosa para la detección temprana de anomalías, permitiendo tomar acciones preventivas y mitigando riesgos. En la monitorización de servidores, podría ayudar a identificar ataques o fallos antes de que causen daños significativos. En sistemas financieros, podría detectar transacciones fraudulentas, protegiendo tanto a las instituciones como a los clientes. Además, el proyecto contribuirá al conocimiento sobre cómo aplicar arquitecturas avanzadas como Transformers en el análisis de series temporales, abriendo oportunidades para futuras investigaciones y aplicaciones.

---

### **Proyecto 7: Optimizar un modelo BERT para dispositivos móviles con técnicas de pruning, quantization y knowledge distillation**

**Descripción detallada:**

El modelo BERT ha demostrado un rendimiento excepcional en diversas tareas de procesamiento del lenguaje natural. Sin embargo, su gran tamaño y complejidad lo hacen poco práctico para su implementación en dispositivos móviles o con recursos limitados. Este proyecto busca reducir el tamaño y mejorar la eficiencia de BERT utilizando técnicas como pruning (poda), quantization (cuantización) y knowledge distillation (destilación de conocimiento), haciendo posible su despliegue en dispositivos móviles.

**Objetivos específicos:**

1. **Reducir el tamaño del modelo BERT sin sacrificar significativamente el rendimiento:** Aplicar técnicas de optimización para disminuir la cantidad de parámetros y el uso de memoria.

2. **Mejorar la eficiencia computacional para dispositivos móviles:** Asegurar que el modelo optimizado pueda ejecutarse con velocidad aceptable en hardware limitado.

3. **Mantener la precisión en tareas específicas:** Evaluar el rendimiento del modelo optimizado en tareas concretas, como clasificación de texto o respuesta a preguntas.

**Metodología:**

- **Análisis del modelo BERT original:**
  - Entender la arquitectura y los componentes clave que contribuyen al rendimiento y al tamaño del modelo.
  - Identificar partes del modelo que pueden ser optimizadas o reducidas.

- **Aplicación de pruning:**
  - Implementar técnicas de poda para eliminar neuronas, capas o conexiones que tengan poca influencia en la salida del modelo.
  - Evaluar el impacto de la poda en el rendimiento y ajustar el nivel de poda para equilibrar tamaño y precisión.

- **Implementación de quantization:**
  - Cuantizar los pesos y activaciones del modelo, reduciendo la precisión numérica (por ejemplo, de 32 bits flotantes a 8 bits enteros).
  - Utilizar técnicas como la cuantización post-entrenamiento o la cuantización consciente del entrenamiento.
  - Asegurar que la cuantización no introduzca errores significativos en las predicciones.

- **Aplicación de knowledge distillation:**
  - Entrenar un modelo más pequeño (estudiante) utilizando el modelo original (maestro) como guía.
  - Transferir el conocimiento del modelo grande al pequeño, intentando mantener la mayor precisión posible.
  - Experimentar con diferentes arquitecturas de modelos estudiantes para encontrar un equilibrio óptimo.

- **Evaluación y pruebas:**
  - Probar el modelo optimizado en dispositivos móviles reales o simulados.
  - Medir métricas como el tiempo de inferencia, uso de memoria y consumo de energía.
  - Comparar el rendimiento con el modelo original y con otros modelos optimizados existentes.

**Consideraciones técnicas:**

- **Compatibilidad con hardware móvil:** Asegurar que el modelo optimizado es compatible con los procesadores y aceleradores comunes en dispositivos móviles, como ARM o GPUs móviles.

- **Frameworks y herramientas:** Utilizar herramientas como TensorFlow Lite o PyTorch Mobile, que facilitan la implementación de modelos en dispositivos móviles y ofrecen soporte para optimizaciones.

- **Precisión vs. eficiencia:** Encontrar el equilibrio adecuado entre la reducción del tamaño y la mantención de la precisión es crucial. Es posible que sea necesario iterar y ajustar las técnicas aplicadas.

**Posibles desafíos:**

- **Degradación del rendimiento:** Las técnicas de optimización pueden causar una disminución significativa en la precisión si no se aplican cuidadosamente.

- **Limitaciones de hardware:** Los dispositivos móviles tienen una variedad de especificaciones, lo que puede complicar la generalización del modelo optimizado.

- **Complejidad de implementación:** Combinar múltiples técnicas de optimización puede aumentar la complejidad del proyecto y requerir un conocimiento profundo del modelo y las herramientas.

**Impacto esperado:**

Al lograr optimizar BERT para dispositivos móviles, se abre la posibilidad de integrar capacidades avanzadas de procesamiento del lenguaje natural en aplicaciones móviles. Esto puede mejorar significativamente la experiencia del usuario en áreas como asistentes virtuales, traducción en tiempo real, análisis de sentimientos y más. Además, el proyecto contribuirá al campo de la optimización de modelos, proporcionando conocimientos y prácticas que pueden aplicarse a otros modelos y contextos.

---

### **Proyecto 8: Crear una aplicación de resumen de documentos largos con un enfoque extractivo y abstractive en LLM**

**Descripción detallada:**

El manejo de grandes volúmenes de información es un desafío común en la era digital. Este proyecto busca desarrollar una aplicación que permita resumir documentos extensos utilizando dos enfoques complementarios: extractivo y abstractive. El enfoque extractivo selecciona las partes más relevantes del texto original, mientras que el enfoque abstractive genera resúmenes que pueden reformular y condensar la información, similar a cómo lo haría un humano.

**Objetivos específicos:**

1. **Implementar un sistema de resumen extractivo efectivo:** Seleccionar automáticamente las oraciones o frases más relevantes de un documento.

2. **Desarrollar un modelo de resumen abstractive utilizando LLM:** Generar resúmenes coherentes y concisos que capturen la esencia del documento.

3. **Crear una aplicación amigable para el usuario:** Permitir que los usuarios carguen documentos y obtengan resúmenes con facilidad.

**Metodología:**

- **Análisis y preprocesamiento de documentos:**
  - Desarrollar métodos para preprocesar los documentos, incluyendo segmentación en oraciones, eliminación de ruido y normalización.

- **Implementación del resumen extractivo:**
  - Utilizar técnicas basadas en estadísticas, como TF-IDF, para identificar las oraciones más significativas.
  - Considerar el uso de algoritmos como TextRank, que aplica métodos de grafos para determinar la importancia de las frases.

- **Desarrollo del modelo de resumen abstractive:**
  - Seleccionar un LLM preentrenado adecuado para la generación de texto, como T5 o BART.
  - Ajustar finamente el modelo utilizando conjuntos de datos de resúmenes, como CNN/Daily Mail.
  - Implementar técnicas para controlar la longitud del resumen y mantener la coherencia.

- **Integración de ambos enfoques en la aplicación:**
  - Permitir que el usuario elija entre resumen extractivo, abstractive o una combinación de ambos.
  - Desarrollar una interfaz que muestre el resumen junto con opciones para ajustar parámetros.

- **Evaluación y mejora del sistema:**
  - Utilizar métricas automáticas como ROUGE para evaluar la calidad de los resúmenes.
  - Recopilar feedback de usuarios para identificar áreas de mejora.

**Consideraciones técnicas:**

- **Limitaciones de los modelos abstractive:** Los modelos pueden generar contenido incorrecto o incoherente. Es importante evaluar cuidadosamente las salidas y ajustar el modelo en consecuencia.

- **Procesamiento de documentos largos:** Los LLMs tienen restricciones en la longitud de entrada. Implementar técnicas como la segmentación del documento y el resumen por partes puede ser necesario.

- **Tiempo de procesamiento:** Generar resúmenes, especialmente abstractive, puede ser computacionalmente intensivo. Optimizar el rendimiento es clave para una buena experiencia de usuario.

**Posibles desafíos:**

- **Calidad y coherencia de los resúmenes:** Garantizar que los resúmenes capturen la esencia del documento sin distorsionar la información.

- **Variedad de formatos de documentos:** Manejar diferentes tipos de documentos y formatos (PDF, Word, texto plano) puede requerir soluciones adicionales.

- **Seguridad y privacidad:** Si los documentos contienen información sensible, es crucial asegurar que los datos estén protegidos y que el procesamiento sea confidencial.

**Impacto esperado:**

La aplicación facilitará el acceso rápido a la información clave en documentos extensos, ahorrando tiempo y esfuerzo a los usuarios. Esto es especialmente útil en entornos académicos, legales, empresariales y de investigación. Al combinar enfoques extractivos y abstractive, se ofrece flexibilidad y se aprovechan las fortalezas de ambos métodos. El proyecto también contribuirá al avance en técnicas de resumen automático y su aplicación práctica.

---

### **Proyecto 9: Ajustar finamente un LLM con datos médicos para tareas como clasificación de documentos clínicos o extracción de información específica**

**Descripción detallada:**

En el sector médico, la gestión y análisis de grandes volúmenes de datos clínicos es un desafío crítico. Este proyecto busca adaptar un modelo de lenguaje preentrenado para procesar información médica, permitiendo la clasificación de documentos clínicos, extracción de datos relevantes y apoyo en la toma de decisiones médicas. Al ajustar finamente el modelo con datos médicos, se busca mejorar su comprensión del lenguaje y terminología específicos del dominio.

**Objetivos específicos:**

1. **Recolectar y preparar un conjunto de datos médicos adecuado:** Asegurar que los datos cumplen con las regulaciones de privacidad y confidencialidad.

2. **Ajustar finamente un LLM para tareas médicas específicas:** Adaptar el modelo para manejar terminología médica y realizar tareas como clasificación y extracción de información.

3. **Evaluar y validar el modelo en entornos reales o simulados:** Asegurar que el modelo es preciso y confiable para su uso en contextos clínicos.

**Metodología:**

- **Recolección y preparación de datos:**
  - Obtener documentos clínicos, registros médicos electrónicos, informes de laboratorio, etc.
  - Anonimizar los datos para proteger la identidad de los pacientes, eliminando información personal identificable.
  - Estructurar y etiquetar los datos según las tareas a realizar (p. ej., categorías de diagnóstico, procedimientos, medicamentos).

- **Cumplimiento de regulaciones y ética:**
  - Asegurar que el uso de los datos cumple con leyes como HIPAA, GDPR u otras aplicables.
  - Obtener aprobaciones éticas y consentimientos necesarios.

- **Ajuste fino del modelo:**
  - Seleccionar un LLM adecuado, como BioBERT o ClinicalBERT, que ya están preentrenados en datos biomédicos.
  - Ajustar el modelo con el conjunto de datos preparado, enfocándose en las tareas específicas.
  - Implementar técnicas para manejar terminología médica y abreviaturas comunes.

- **Evaluación del modelo:**
  - Utilizar métricas relevantes para las tareas, como precisión, recall y F1-score.
  - Comparar el rendimiento con modelos base o métodos tradicionales.

- **Implementación y pruebas:**
  - Desarrollar prototipos o herramientas que integren el modelo, como sistemas de ayuda a la decisión clínica o asistentes para médicos.
  - Realizar pruebas en entornos controlados con profesionales médicos, recopilando feedback.

**Consideraciones técnicas:**

- **Complejidad del lenguaje médico:** El lenguaje clínico es altamente especializado y puede incluir términos raros, acrónimos y variaciones lingüísticas.

- **Balance de datos:** Puede haber clases desbalanceadas (p. ej., ciertas enfermedades son más comunes que otras), lo que requiere técnicas para manejar el desequilibrio.

- **Interpretabilidad y explicabilidad:** En contextos médicos, es crucial que las decisiones del modelo sean interpretables y justificables.

**Posibles desafíos:**

- **Privacidad y seguridad de los datos:** Manejar datos médicos requiere estrictas medidas de seguridad para prevenir filtraciones o usos indebidos.

- **Responsabilidad legal y ética:** Las recomendaciones o clasificaciones erróneas pueden tener consecuencias graves. Es importante establecer límites claros sobre el uso del modelo y garantizar supervisión humana.

- **Actualización constante:** El campo médico evoluciona rápidamente, por lo que el modelo debe actualizarse regularmente para incorporar nuevos conocimientos y prácticas.

**Impacto esperado:**

La implementación de este proyecto puede mejorar significativamente la eficiencia en la gestión de información clínica, apoyar a los profesionales de la salud en la toma de decisiones y contribuir a mejores resultados para los pacientes. Además, el proyecto avanzará en la aplicación de LLMs en el sector médico, demostrando cómo pueden adaptarse y utilizarse de manera segura y efectiva en dominios especializados.

---

### **Proyecto 10: Crear un sistema de generación de texto condicional basado en estilos o temas específicos usando técnicas de control de generación en LLMs**

**Descripción detallada:**

La generación de texto condicional permite controlar las características del texto generado por un modelo de lenguaje, como el estilo, el tono o el tema. Este proyecto busca implementar un sistema que pueda generar texto adaptado a estilos o temas específicos, utilizando técnicas de control de generación en LLMs. Esto tiene aplicaciones en áreas como la creación de contenido personalizado, asistentes de escritura y generación de diálogo en personajes de videojuegos.

**Objetivos específicos:**

1. **Recolectar y preparar conjuntos de datos representativos de diferentes estilos o temas:** Asegurar que los datos reflejen las características deseadas.

2. **Implementar técnicas de control de generación en LLMs:** Adaptar el modelo para que pueda generar texto condicionado por parámetros específicos.

3. **Desarrollar una interfaz que permita a los usuarios especificar condiciones y generar texto:** Facilitar la interacción con el sistema.

**Metodología:**

- **Recolección y etiquetado de datos:**
  - Reunir textos que representen los estilos o temas deseados (p. ej., formal, coloquial, humorístico, científico).
  - Etiquetar los datos con las características correspondientes.

- **Selección y ajuste del modelo:**
  - Elegir un LLM adecuado, como GPT-2 o GPT-3, que tenga capacidad para generación de texto.
  - Implementar técnicas de control, como el uso de prompts, etiquetas explícitas o embeddings de control.
  - Ajustar finamente el modelo utilizando los datos etiquetados.

- **Implementación de técnicas de control:**
  - **Control con etiquetas:** Incorporar etiquetas al inicio del texto que indiquen el estilo o tema deseado.
  - **Control mediante prompts:** Diseñar prompts que guíen al modelo hacia el estilo o tema específico.
  - **Control mediante embeddings:** Utilizar vectores de control que influyan en la generación del modelo.

- **Desarrollo de la interfaz de usuario:**
  - Crear una aplicación que permita a los usuarios seleccionar condiciones (p. ej., estilo, tema) y generar texto.
  - Implementar opciones para ajustar la longitud, el tono y otros parámetros.

- **Evaluación y refinamiento:**
  - Evaluar la calidad del texto generado mediante métricas automáticas y feedback de usuarios.
  - Realizar ajustes en el modelo y las técnicas de control para mejorar la coherencia y adecuación.

**Consideraciones técnicas:**

- **Evitar sesgos y contenido inapropiado:** Los modelos pueden generar contenido sesgado o inapropiado. Es importante implementar filtros y supervisión.

- **Diversidad y creatividad:** Equilibrar la coherencia con la creatividad, evitando que el texto sea repetitivo o predecible.

- **Limitaciones de los modelos:** Los LLMs pueden tener dificultades para mantener el contexto a largo plazo o para adherirse estrictamente a las condiciones establecidas.

**Posibles desafíos:**

- **Control preciso del estilo:** Conseguir que el modelo genere texto que cumpla exactamente con las condiciones puede ser complejo.

- **Evaluación subjetiva:** La calidad y adecuación del texto pueden ser subjetivas y variar entre usuarios.

- **Recursos computacionales:** El ajuste fino y la generación de texto pueden requerir recursos significativos, especialmente con modelos grandes.

**Impacto esperado:**

Este proyecto permitirá generar texto personalizado y adaptado a necesidades específicas, lo que puede ser de gran utilidad en marketing, educación, entretenimiento y más. Al facilitar la creación de contenido de alta calidad y alineado con ciertos estilos o temas, se pueden mejorar procesos creativos y comunicativos. Además, el proyecto contribuirá al desarrollo de técnicas de control de generación en LLMs, un área de creciente interés en el campo de la inteligencia artificial.

---

### **Proyecto 11: Optimizar un modelo Transformer con técnicas de pruning y quantization para su despliegue en dispositivos edge**

**Descripción detallada:**

Los dispositivos edge, como sensores inteligentes, dispositivos IoT y sistemas embebidos, suelen tener recursos limitados en términos de computación y energía. Este proyecto tiene como objetivo optimizar un modelo Transformer para que pueda ser desplegado en estos dispositivos, utilizando técnicas de pruning y quantization para reducir su tamaño y mejorar la eficiencia sin sacrificar significativamente el rendimiento.

**Objetivos específicos:**

1. **Reducir el tamaño y la complejidad del modelo Transformer:** Aplicar pruning para eliminar parámetros redundantes o menos significativos.

2. **Mejorar la eficiencia computacional mediante quantization:** Reducir la precisión numérica de los parámetros para disminuir el uso de memoria y acelerar la inferencia.

3. **Desplegar y evaluar el modelo en dispositivos edge reales:** Asegurar que el modelo funciona correctamente en el entorno objetivo y cumple con los requisitos de rendimiento.

**Metodología:**

- **Análisis del modelo original:**
  - Comprender la arquitectura del Transformer y determinar las partes que consumen más recursos.
  - Identificar oportunidades para reducir la complejidad sin afectar la precisión.

- **Aplicación de pruning:**
  - Implementar técnicas como pruning basado en magnitud, que elimina conexiones con pesos pequeños.
  - Experimentar con diferentes niveles de pruning y evaluar el impacto en el rendimiento.

- **Implementación de quantization:**
  - Reducir la precisión de los pesos y activaciones del modelo, por ejemplo, de 32 bits flotantes a 8 bits enteros.
  - Utilizar quantization estática o dinámica según sea apropiado.

- **Ajuste fino del modelo optimizado:**
  - Retrain el modelo después de la optimización para recuperar parte de la precisión perdida.
  - Ajustar hiperparámetros y aplicar técnicas de regularización si es necesario.

- **Despliegue en dispositivos edge:**
  - Utilizar frameworks como TensorFlow Lite, ONNX Runtime o PyTorch Mobile que soportan optimizaciones y despliegue en dispositivos edge.
  - Implementar el modelo en un dispositivo real y realizar pruebas de rendimiento.

- **Evaluación y ajustes:**
  - Medir métricas como tiempo de inferencia, uso de memoria, consumo de energía y precisión.
  - Comparar con el modelo original y con otros modelos optimizados.

**Consideraciones técnicas:**

- **Compatibilidad con hardware:** Asegurar que las optimizaciones son compatibles con el hardware específico del dispositivo edge.

- **Limitaciones de memoria y energía:** Los dispositivos edge pueden tener restricciones severas, lo que requiere una optimización agresiva.

- **Seguridad y confiabilidad:** En algunos casos, los dispositivos edge operan en entornos críticos, por lo que el modelo debe ser confiable y seguro.

**Posibles desafíos:**

- **Degradación de la precisión:** Las técnicas de pruning y quantization pueden reducir la precisión del modelo. Es crucial encontrar un equilibrio adecuado.

- **Herramientas limitadas:** Las herramientas y frameworks para optimizar y desplegar modelos en dispositivos edge pueden ser limitadas o menos maduras.

- **Variabilidad de dispositivos:** Los dispositivos edge son diversos, lo que puede dificultar la creación de una solución generalizada.

**Impacto esperado:**

Optimizar un modelo Transformer para dispositivos edge permite llevar capacidades avanzadas de inteligencia artificial a entornos donde antes no era posible, habilitando aplicaciones como análisis en tiempo real, detección de anomalías, reconocimiento de patrones y más, directamente en el dispositivo. Esto reduce la necesidad de comunicación con servidores remotos, mejora la latencia y protege la privacidad de los datos. El proyecto también contribuirá al avance de técnicas de optimización de modelos y su aplicación práctica en entornos con recursos limitados.

---
## Rúbricas

## **Proyecto 1: Ajustar finamente un LLM con PPO vs DPO vs ORPO utilizando el paquete PEFT**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                   | Puntos |
|------------------------------------------------------------|--------|
| Preparación y configuración del entorno de desarrollo      | 2      |
| Selección y justificación del conjunto de datos adecuado   | 2      |
| Configuración inicial del LLM base para el ajuste fino     | 2      |
| Documentación clara y detallada de los pasos realizados    | 2      |
| **Total**                                                  | **8**  |

**Exposición (12 puntos)**

| Criterio                                                   | Puntos |
|------------------------------------------------------------|--------|
| Presentación clara de los objetivos del entregable         | 3      |
| Explicación de la configuración del entorno y herramientas | 3      |
| Uso de recursos visuales (diapositivas, gráficos)          | 3      |
| Respuesta efectiva a preguntas y manejo del tiempo         | 3      |
| **Total**                                                  | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                       | Puntos |
|----------------------------------------------------------------|--------|
| Implementación correcta de PPO, DPO y ORPO utilizando PEFT     | 3      |
| Entrenamiento exitoso del LLM con cada método                  | 3      |
| Registro y análisis de métricas de rendimiento                 | 2      |
| **Total**                                                      | **8**  |

**Exposición (12 puntos)**

| Criterio                                                           | Puntos |
|--------------------------------------------------------------------|--------|
| Explicación detallada de cada método de optimización               | 4      |
| Demostración de resultados intermedios y desafíos enfrentados      | 4      |
| Claridad en la comunicación y uso de ejemplos prácticos            | 4      |
| **Total**                                                          | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                        | Puntos |
|-----------------------------------------------------------------|--------|
| Evaluación exhaustiva del rendimiento de cada modelo ajustado   | 3      |
| Análisis comparativo detallado entre PPO, DPO y ORPO            | 3      |
| Informe final con conclusiones y recomendaciones                | 2      |
| **Total**                                                       | **8**  |

**Exposición (12 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Presentación de resultados y hallazgos clave                         | 4      |
| Análisis crítico y reflexión sobre el proceso y resultados           | 4      |
| Calidad visual y estructuración de la presentación                   | 4      |
| **Total**                                                            | **12** |

---

## **Proyecto 2: Crear una aplicación interactiva de chat que utiliza GPT para responder en tiempo real, con soporte para WebSockets**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                      | Puntos |
|---------------------------------------------------------------|--------|
| Desarrollo y configuración del backend del servidor           | 3      |
| Integración exitosa con la API de GPT preentrenado            | 3      |
| Implementación de endpoints básicos para el chat              | 2      |
| **Total**                                                     | **8**  |

**Exposición (12 puntos)**

| Criterio                                                         | Puntos |
|------------------------------------------------------------------|--------|
| Descripción clara de la arquitectura del backend                 | 4      |
| Demostración de funcionalidades implementadas                    | 4      |
| Respuesta a preguntas técnicas y manejo del tiempo               | 4      |
| **Total**                                                        | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                      | Puntos |
|---------------------------------------------------------------|--------|
| Desarrollo de la interfaz de usuario del chat                 | 3      |
| Implementación efectiva de WebSockets para comunicación       | 3      |
| Pruebas de interacción entre frontend y backend               | 2      |
| **Total**                                                     | **8**  |

**Exposición (12 puntos)**

| Criterio                                                            | Puntos |
|---------------------------------------------------------------------|--------|
| Presentación del frontend y sus características                     | 4      |
| Explicación del uso de WebSockets y su importancia                  | 4      |
| Demostración en vivo de la aplicación en funcionamiento             | 4      |
| **Total**                                                           | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                         | Puntos |
|------------------------------------------------------------------|--------|
| Optimización de la aplicación para rendimiento y escalabilidad   | 3      |
| Implementación de medidas de seguridad y manejo de errores       | 3      |
| Despliegue exitoso de la aplicación y documentación final        | 2      |
| **Total**                                                        | **8**  |

**Exposición (12 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Presentación de mejoras y optimizaciones realizadas                   | 4      |
| Discusión sobre desafíos enfrentados y soluciones implementadas       | 4      |
| Calidad general de la exposición y uso de recursos visuales           | 4      |
| **Total**                                                             | **12** |

---

## **Proyecto 3: Entrenar y ajustar un LLM especializado en la clasificación de noticias por temas**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                          | Puntos |
|-------------------------------------------------------------------|--------|
| Recolección y preprocesamiento adecuado del conjunto de datos     | 3      |
| Etiquetado correcto de las categorías temáticas                   | 3      |
| Selección y justificación del modelo base (e.g., DistilBERT)      | 2      |
| **Total**                                                         | **8**  |

**Exposición (12 puntos)**

| Criterio                                                          | Puntos |
|-------------------------------------------------------------------|--------|
| Explicación del proceso de recopilación y preparación de datos    | 4      |
| Justificación de las decisiones tomadas en la selección del modelo| 4      |
| Claridad en la presentación y respuesta a preguntas               | 4      |
| **Total**                                                         | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                      | Puntos |
|---------------------------------------------------------------|--------|
| Configuración y ajuste fino del modelo para la tarea          | 3      |
| Entrenamiento y validación del modelo con resultados iniciales| 3      |
| Ajuste de hiperparámetros y mejoras implementadas             | 2      |
| **Total**                                                     | **8**  |

**Exposición (12 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Presentación de la metodología de entrenamiento                      | 4      |
| Análisis de resultados y discusión de mejoras                        | 4      |
| Uso efectivo de gráficos y tablas para ilustrar puntos clave         | 4      |
| **Total**                                                            | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                          | Puntos |
|-------------------------------------------------------------------|--------|
| Evaluación del modelo en conjunto de pruebas independiente        | 3      |
| Desarrollo de interfaz o API funcional                            | 3      |
| Documentación completa y propuestas de mejoras futuras            | 2      |
| **Total**                                                         | **8**  |

**Exposición (12 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Demostración de la herramienta o API desarrollada                    | 4      |
| Evaluación final y reflexión sobre el proyecto                       | 4      |
| Calidad de la presentación y habilidad para comunicar ideas          | 4      |
| **Total**                                                            | **12** |

---

## **Proyecto 4: Usar embeddings contextuales para generar recomendaciones de películas**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Creación y preprocesamiento del conjunto de datos de películas       | 3      |
| Generación correcta de embeddings para descripciones                 | 3      |
| Organización y almacenamiento eficiente de los embeddings            | 2      |
| **Total**                                                            | **8**  |

**Exposición (12 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Explicación del proceso de generación de embeddings                  | 4      |
| Presentación de desafíos y soluciones en la preparación de datos     | 4      |
| Claridad y coherencia en la exposición                               | 4      |
| **Total**                                                            | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Desarrollo de método para capturar preferencias del usuario            | 3      |
| Conversión de preferencias en embeddings compatibles                   | 3      |
| Implementación del algoritmo de similitud entre embeddings             | 2      |
| **Total**                                                              | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                  | Puntos |
|---------------------------------------------------------------------------|--------|
| Demostración de cómo se capturan y procesan las preferencias del usuario  | 4      |
| Explicación del algoritmo de recomendación basado en similitud            | 4      |
| Interacción con la audiencia y manejo de preguntas                        | 4      |
| **Total**                                                                 | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Integración completa del sistema de recomendación                      | 3      |
| Pruebas realizadas y ajustes finales implementados                     | 3      |
| Documentación detallada del sistema y su uso                           | 2      |
| **Total**                                                              | **8**  |

**Exposición (12 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Demostración en vivo del sistema funcionando                           | 4      |
| Presentación de resultados y feedback obtenido                         | 4      |
| Calidad general de la exposición y recursos visuales utilizados        | 4      |
| **Total**                                                              | **12** |

---

## **Proyecto 5: Desplegar una API de preguntas y respuestas basada en LLM finamente ajustado**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                                | Puntos |
|-------------------------------------------------------------------------|--------|
| Ajuste fino del LLM para preguntas y respuestas                         | 3      |
| Indexación y organización eficiente de la base de datos de documentos   | 3      |
| Implementación inicial de búsqueda eficiente en los documentos          | 2      |
| **Total**                                                               | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Descripción del LLM y su ajuste para la tarea específica                 | 4      |
| Explicación de la base de datos y mecanismos de búsqueda implementados   | 4      |
| Claridad en la presentación y capacidad para responder preguntas         | 4      |
| **Total**                                                                | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Desarrollo de la API con endpoints funcionales                       | 3      |
| Integración del LLM con la búsqueda de documentos para respuestas    | 3      |
| Manejo de excepciones y validación de entradas de usuario            | 2      |
| **Total**                                                            | **8**  |

**Exposición (12 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Demostración del funcionamiento de la API                             | 4      |
| Explicación de la lógica detrás de la generación de respuestas        | 4      |
| Interacción efectiva con la audiencia y manejo del tiempo             | 4      |
| **Total**                                                             | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                            | Puntos |
|---------------------------------------------------------------------|--------|
| Pruebas exhaustivas realizadas y optimización de la API             | 3      |
| Documentación completa y ejemplos de uso para desarrolladores       | 3      |
| Reflexión sobre posibles mejoras y futuras expansiones              | 2      |
| **Total**                                                           | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                | Puntos |
|-------------------------------------------------------------------------|--------|
| Presentación de resultados finales y desempeño de la API                | 4      |
| Discusión de desafíos enfrentados y cómo se superaron                   | 4      |
| Calidad visual y estructuración de la presentación                      | 4      |
| **Total**                                                               | **12** |

---

## **Proyecto 6: Utilizar Transformers para detectar anomalías en datos de series temporales**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Recolección y preprocesamiento adecuado de los datos de series temporales | 3      |
| Análisis exploratorio y identificación de patrones                    | 3      |
| Configuración inicial del modelo Transformer adaptado                 | 2      |
| **Total**                                                             | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Explicación de la importancia y desafíos de detectar anomalías           | 4      |
| Presentación del análisis de datos y hallazgos iniciales                 | 4      |
| Claridad y efectividad en la comunicación                                | 4      |
| **Total**                                                                | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Entrenamiento del modelo con datos históricos                          | 3      |
| Validación y ajuste de hiperparámetros                                 | 3      |
| Implementación de técnicas para prevenir sobreajuste                   | 2      |
| **Total**                                                              | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Explicación del proceso de entrenamiento y validación                    | 4      |
| Presentación de resultados intermedios y métricas                        | 4      |
| Interacción con la audiencia y capacidad para responder preguntas        | 4      |
| **Total**                                                                | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Desarrollo de herramienta para aplicación en tiempo real             | 3      |
| Pruebas realizadas en entornos simulados o reales                    | 3      |
| Documentación y propuestas para integración futura                   | 2      |
| **Total**                                                            | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Demostración del modelo funcionando en tiempo real                       | 4      |
| Análisis de resultados y discusión sobre el impacto del proyecto         | 4      |
| Calidad general de la presentación y uso de recursos visuales            | 4      |
| **Total**                                                                | **12** |

---

## **Proyecto 7: Optimizar un modelo BERT para dispositivos móviles**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                          | Puntos |
|-------------------------------------------------------------------|--------|
| Análisis y aplicación de pruning al modelo BERT                   | 3      |
| Evaluación del impacto del pruning en tamaño y precisión          | 3      |
| Ajuste fino post-pruning para mejorar rendimiento                 | 2      |
| **Total**                                                         | **8**  |

**Exposición (12 puntos)**

| Criterio                                                            | Puntos |
|---------------------------------------------------------------------|--------|
| Explicación detallada del proceso de pruning                        | 4      |
| Presentación de resultados y comparación con el modelo original     | 4      |
| Claridad en la comunicación y manejo del tiempo                     | 4      |
| **Total**                                                           | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                           | Puntos |
|--------------------------------------------------------------------|--------|
| Implementación de quantization en el modelo podado                  | 3      |
| Pruebas de inferencia en dispositivos limitados                     | 3      |
| Resolución de problemas de precisión causados por la quantization   | 2      |
| **Total**                                                           | **8**  |

**Exposición (12 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Explicación de la técnica de quantization y su importancia             | 4      |
| Demostración de mejoras en eficiencia y discusiones sobre desafíos     | 4      |
| Interacción efectiva con la audiencia                                  | 4      |
| **Total**                                                              | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Aplicación de knowledge distillation y entrenamiento del modelo      | 3      |
| Comparación del modelo distilado con el original                     | 3      |
| Despliegue y evaluación en dispositivo móvil                         | 2      |
| **Total**                                                            | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                  | Puntos |
|---------------------------------------------------------------------------|--------|
| Presentación de resultados finales y rendimiento en dispositivos móviles  | 4      |
| Reflexión sobre el proceso y aprendizajes obtenidos                       | 4      |
| Calidad de la exposición y uso de recursos visuales                       | 4      |
| **Total**                                                                 | **12** |

---

## **Proyecto 8: Crear una aplicación de resumen de documentos largos**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Implementación del enfoque de resumen extractivo                       | 3      |
| Utilización de modelos preentrenados para evaluar importancia          | 3      |
| Pruebas iniciales con documentos de ejemplo                            | 2      |
| **Total**                                                              | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                    | Puntos |
|-----------------------------------------------------------------------------|--------|
| Explicación del enfoque extractivo y su implementación                      | 4      |
| Demostración de resultados iniciales                                        | 4      |
| Claridad y coherencia en la presentación                                    | 4      |
| **Total**                                                                   | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                                | Puntos |
|-------------------------------------------------------------------------|--------|
| Implementación del enfoque de resumen abstractive                       | 3      |
| Ajuste fino del modelo para mejorar coherencia y fluidez                | 3      |
| Comparación de resultados entre enfoques extractivo y abstractive       | 2      |
| **Total**                                                               | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Explicación del enfoque abstractive y desafíos asociados                 | 4      |
| Presentación comparativa de ambos enfoques                               | 4      |
| Interacción con la audiencia y capacidad para responder preguntas        | 4      |
| **Total**                                                                | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Desarrollo de la aplicación con opciones para ambos tipos de resumen  | 3      |
| Evaluación con usuarios y ajustes realizados según feedback           | 3      |
| Documentación y preparación para despliegue                           | 2      |
| **Total**                                                             | **8**  |

**Exposición (12 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Demostración de la aplicación funcionando y sus características       | 4      |
| Discusión sobre el impacto y posibles aplicaciones                    | 4      |
| Calidad general de la presentación y recursos utilizados              | 4      |
| **Total**                                                             | **12** |

---

## **Proyecto 9: Ajustar finamente un LLM con datos médicos**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Recolección de datos médicos cumpliendo regulaciones                 | 3      |
| Anonimización y preprocesamiento adecuado de los datos               | 3      |
| Revisión ética y legal del uso de los datos                          | 2      |
| **Total**                                                            | **8**  |

**Exposición (12 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Explicación de los procedimientos para asegurar privacidad y ética     | 4      |
| Presentación de los datos y su relevancia para el proyecto             | 4      |
| Claridad en la exposición y manejo de preguntas                        | 4      |
| **Total**                                                              | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                             | Puntos |
|----------------------------------------------------------------------|--------|
| Configuración y entrenamiento del LLM para tareas médicas            | 3      |
| Validación del modelo en tareas de clasificación o extracción         | 3      |
| Ajustes realizados para mejorar el rendimiento                        | 2      |
| **Total**                                                            | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Presentación del modelo y su adaptación al dominio médico                | 4      |
| Resultados obtenidos y análisis de desempeño                             | 4      |
| Interacción efectiva con la audiencia                                    | 4      |
| **Total**                                                                | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                                | Puntos |
|-------------------------------------------------------------------------|--------|
| Desarrollo de herramienta o aplicación que utiliza el modelo            | 3      |
| Pruebas realizadas con profesionales médicos y feedback recopilado      | 3      |
| Documentación y reflexión sobre mejoras futuras                         | 2      |
| **Total**                                                               | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Demostración de la herramienta en funcionamiento                         | 4      |
| Discusión sobre el impacto y consideraciones éticas                      | 4      |
| Calidad de la presentación y recursos visuales utilizados                | 4      |
| **Total**                                                                | **12** |

---

## **Proyecto 10: Crear un sistema de generación de texto condicional basado en estilos o temas específicos**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Recolección y etiquetado de datos representativos de estilos/temas     | 3      |
| Configuración del LLM para aceptar condiciones o controles             | 3      |
| Preprocesamiento y preparación de los datos para entrenamiento         | 2      |
| **Total**                                                              | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Explicación de la importancia del control en generación de texto         | 4      |
| Presentación de los datos y su relevancia                                | 4      |
| Claridad y efectividad en la comunicación                                | 4      |
| **Total**                                                                | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Entrenamiento del modelo con técnicas de control implementadas         | 3      |
| Pruebas de generación de texto en diferentes estilos/temas             | 3      |
| Ajuste de parámetros para mejorar calidad y coherencia                  | 2      |
| **Total**                                                              | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                 | Puntos |
|--------------------------------------------------------------------------|--------|
| Demostración de ejemplos de texto generado bajo diferentes condiciones   | 4      |
| Explicación de las técnicas de control utilizadas                        | 4      |
| Interacción con la audiencia y capacidad para responder preguntas        | 4      |
| **Total**                                                                | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Desarrollo de la interfaz que permite seleccionar condiciones          | 3      |
| Evaluación del sistema con usuarios y recopilación de feedback         | 3      |
| Refinamiento final y documentación del sistema                         | 2      |
| **Total**                                                              | **8**  |

**Exposición (12 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Presentación de la interfaz y funcionalidades del sistema              | 4      |
| Discusión sobre resultados y posibles aplicaciones prácticas           | 4      |
| Calidad general de la exposición y uso de recursos visuales            | 4      |
| **Total**                                                              | **12** |

---

## **Proyecto 11: Optimizar un modelo Transformer con técnicas de pruning y quantization para dispositivos edge**

### **Entregable 1 - Fecha: 30 de noviembre**

**Trabajo (8 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Identificación y aplicación de pruning al modelo Transformer          | 3      |
| Evaluación del impacto en rendimiento y tamaño del modelo             | 3      |
| Retraining del modelo post-pruning si es necesario                    | 2      |
| **Total**                                                             | **8**  |

**Exposición (12 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Explicación de la necesidad de optimización para dispositivos edge     | 4      |
| Presentación del proceso y resultados del pruning                      | 4      |
| Claridad y efectividad en la comunicación                              | 4      |
| **Total**                                                              | **12** |

### **Entregable 2 - Fecha: 7 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                              | Puntos |
|-----------------------------------------------------------------------|--------|
| Implementación de quantization y optimización adicional               | 3      |
| Pruebas de rendimiento en simuladores de dispositivos edge            | 3      |
| Optimización de velocidad de inferencia y uso de memoria              | 2      |
| **Total**                                                             | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                  | Puntos |
|---------------------------------------------------------------------------|--------|
| Explicación de las técnicas de quantization y su aplicación               | 4      |
| Demostración de mejoras en rendimiento y eficiencia                       | 4      |
| Interacción efectiva con la audiencia                                     | 4      |
| **Total**                                                                 | **12** |

### **Entregable 3 - Fecha: 14 de diciembre**

**Trabajo (8 puntos)**

| Criterio                                                               | Puntos |
|------------------------------------------------------------------------|--------|
| Despliegue del modelo optimizado en un dispositivo edge                | 3      |
| Evaluación del rendimiento en condiciones reales                       | 3      |
| Documentación de procedimientos y resultados                           | 2      |
| **Total**                                                              | **8**  |

**Exposición (12 puntos)**

| Criterio                                                                  | Puntos |
|---------------------------------------------------------------------------|--------|
| Demostración en vivo del modelo funcionando en el dispositivo edge        | 4      |
| Discusión sobre los desafíos y soluciones implementadas                   | 4      |
| Calidad general de la presentación y uso de recursos visuales             | 4      |
| **Total**                                                                 | **12** |


