## **Introducción a las técnicas de fine-tuning**

En el contexto de los modelos de lenguaje de gran escala (LLMs), el fine-tuning es el proceso de ajustar un modelo preentrenado con grandes volúmenes de  datos para que desempeñe tareas específicas (por ejemplo, clasificación de texto, generación de código, diálogo, etc.). 
Dado el tamaño masivo de estos modelos (desde decenas hasta centenas de miles de millones de parámetros), el reto principal radica en equilibrar la eficacia
del ajuste con los recursos computacionales disponibles. 

Mientras que el "full fine-tuning" ajustar todos los parámetros del modelo, proporciona la mayor flexibilidad, también conlleva costos de cómputo, memoria y almacenamiento muy elevados. 

Para mitigar estos desafíos, se han desarrollado técnicas de **Parameter-Efficient Fine-Tuning (PEFT)** como **LoRA** y sus variantes, así como metodologías de alineación basadas en retroalimentación humana, como **RLHF**.

### **1. Full fine-tuning**

Full fine-tuning implica actualizar todos los pesos del modelo preentrenado durante la fase de ajuste. 

En la práctica, se inicializa un bucle de entrenamiento completo (forward y backward pass) sobre el conjunto de datos específico de la tarea y se aplica un optimizador (por ejemplo, Adam) para modificar cada uno de los parámetros del transformer.

**Ventajas:**

* Flexibilidad máxima: al ajustar cada parámetro, el modelo puede adaptar sus representaciones internas de forma óptima para la nueva tarea.
* Rendimiento líder: en problemas complejos, full fine-tuning suele superar a métodos más ligeros.

**Limitaciones:**

* Costos de memoria muy elevados: por ejemplo, ajustar un modelo con cientos de miles de millones de parámetros puede requerir cientos de gigabytes de VRAM.
* Tiempo de entrenamiento prolongado: puede tomar días o semanas incluso en clusters de GPU de alta gama.
* Almacenamiento de checkpoints: cada punto de control ocupa un espacio significativo en disco.

**Casos de uso recomendados:**

Full fine-tuning se reserva para escenarios en los que se dispone de infraestructura de última generación y donde la precisión absoluta es crítica.

### **2. Low-Rank Adaptation (LoRA) y QLoRA**

LoRA (Low-Rank Adaptation) es una técnica de fine-tuning eficiente en parámetros. Consiste en congelar los pesos originales del modelo y añadir matrices de bajo rango A y B en cada capa de atención. 
Solo estas matrices se actualizan durante el entrenamiento, reduciendo drásticamente el número de parámetros entrenables.

**Beneficios de LoRA:**

* Reducción de parámetros entrenables en varios órdenes de magnitud.
* Menor uso de memoria y ancho de banda en entornos distribuidos.
* Checkpoints mucho más ligeros.

QLoRA extiende LoRA añadiendo cuantización de los pesos preentrenados a 4 bits, mientras mantiene las matrices de adaptación en 16 bits. Esto permite:

* Disminuir aún más el uso de VRAM (por ejemplo, de alrededor de 20 GB a 14 GB).
* Mantener la calidad de generación casi idéntica.
* Seguir entrenando modelos de decenas de miles de millones de parámetros en GPUs de consumo.

### **3 . Supervised fine-tuning (SFT)**

Supervised Fine-Tuning consiste en entrenar un LLM con un conjunto de datos etiquetado (entrada -> salida esperada). El flujo de trabajo típico incluye:

1. Construcción del dataset con ejemplos de prompt y respuesta.
2. Ajuste del modelo mediante mínimos cuadrados o cross-entropy.
3. Validación para evitar sobreajuste.

**Ventajas:**

* Sencillez de implementación con frameworks estándar (PyTorch, TensorFlow).
* Control directo de la calidad de salida mediante datos de demostración.
* Versatilidad para cualquier tarea formulable como input-output.

**Desafíos:**

* Dependencia de la disponibilidad y calidad del corpus etiquetado.
* Riesgo de sesgos o sobreajuste a estilos presentes en los datos.
* Limitaciones de generalización a situaciones no vistas en los ejemplos.

### **4. Reinforcement Learning from Human Feedback (RLHF)**

RLHF surge para alinear las salidas de un LLM con las preferencias humanas. El proceso consta de tres fases:

1. Recolección de datos de preferencia: los anotadores comparan varias salidas del modelo y eligen la preferida.
2. Entrenamiento de un Reward Model (RM) que asigna un puntaje a cada respuesta.
3. Optimización del policy (el LLM) mediante un algoritmo de RL como PPO, maximizando la recompensa predicha por el RM.

#### **Variantes:**

* **Direct Preference Optimization (DPO)**, que optimiza directamente la probabilidad de las respuestas preferidas sin entrenar un modelo de recompensa intermedio.
* **Reinforcement Learning from AI Feedback (RLAIF)**, que usa otro LLM para generar comparaciones de preferencia, aumentando la escalabilidad a costa de potenciales sesgos.


El RLHF (Reinforcement Learning from Human Feedback) busca alinear las respuestas del modelo con las preferencias reales de los usuarios. El algoritmo más común para esa última fase es **PPO** (Proximal Policy Optimization), que actualiza gradualmente los parámetros sin desviarse demasiado de la política original, lo que ayuda a mantener el entrenamiento estable y controlado.

Una variante más sencilla es **DPO** (Direct Preference Optimization). En lugar de entrenar un reward model por separado, DPO trabaja directamente sobre los ejemplos comparativos: dada una respuesta preferida y otra no preferida, el modelo aprende a asignar más probabilidad a la primera de forma directa. 

Esto elimina el componente intermedio de recompensa y hace el pipeline más ligero, reduciendo complejidad y puntos de fallo, aunque puede perder algo de flexibilidad para representar matices muy sutiles en la preferencia humana.

Entre ambas aproximaciones, PPO ofrece una alineación más fina y ha demostrado ser robusta para tareas de diálogo o generación creativa, pero conlleva mayor complejidad operativa (buffers de experiencia, más fases de entrenamiento).
DPO, por su parte, es más rápida de desplegar y ocupa menos espacio de almacenamiento, siendo ideal cuando buscamos un buen alineamiento sin montar una infraestructura de RL completa.

**Beneficios de RLHF y sus variantes:**

* Mejor alineación continua con criterios humanos: tono, seguridad, relevancia.
* Reducción de alucinaciones y salida de información incorrecta.
* Flexibilidad para definir distintas métricas de recompensa.

**Retos:**

* Alto costo de anotación humana.
* Inestabilidad de entrenamiento al combinar RL con redes neuronales grandes.
* Riesgo de amplificar sesgos presentes en los datos de preferencia.


### **5. Adapters**

Los **adapters** son una técnica de fine-tuning muy parecida a LoRA, pensada para añadir solo unos pocos módulos ligeros entre las capas de un transformer, en lugar de ajustar todo el modelo. 

Cada adapter consiste en una pequeña "expansión" y "compresión" de la representación interna: primero reduce la dimensión a un espacio más pequeño, aplica una función no lineal y luego vuelve a expandirla al tamaño original. 

Durante el entrenamiento, todos los pesos originales del modelo permanecen congelados, y únicamente se ajustan estos módulos extra.  Gracias a ello, los checkpoints ocupan apenas un 1-5 % del tamaño del modelo completo, lo que facilita almacenar, compartir y cargar adaptaciones para tareas 
muy distintas sin interferencias entre ellas. 

Su principal ventaja es su modularidad: puedes tener varios adapters para diferentes dominios y activarlos según la tarea. El pequeño coste adicional en cómputo durante la fase de inferencia se ve compensado por el ahorro de memoria y la rapidez al cambiar de contexto. 
Se suelen usar cuando necesitamos soporte multi-tarea o disponemos de recursos muy limitados para guardar adaptaciones pesadas.


En conjunto, el ecosistema actual de fine-tuning nos brinda múltiples palancas para equilibrar rendimiento, coste y flexibilidad:

* **Full fine-tuning** sigue siendo la opción de máximo rendimiento cuando disponemos de infraestructuras de alto nivel, pero es la más cara en memoria y tiempo de entrenamiento.
* **LoRA** y **QLoRA** permitieron democratizar el ajuste de modelos grandes gracias a sus matrices de bajo rango y la cuantización de parámetros, reduciendo notablemente la necesidad de VRAM sin sacrificar calidad.
* **Adapters** amplían esa idea incorporando módulos intercambiables que facilitan escenarios multi-tarea y cambios de dominio instantáneos.
* **SFT** sobre datos etiquetados sigue siendo muy útil cuando contamos con buenos ejemplos y necesitamos un control absoluto sobre el estilo de salida.
* **RLHF con PPO** aporta una afinación continua según criterios humanos, mejorando el tono y la seguridad de los sistemas conversacionales.
* **DPO** y **RLAIF** reducen aún más la complejidad de RLHF al evitar o simplificar el entrenamiento de modelos de recompensa, manteniendo buena alineación con menor gasto de anotación y de infraestructura.

### **Consideraciones prácticas**

* Infraestructura y presupuesto:
  * Con GPUs de gran capacidad y largo plazo, full fine-tuning es viable.
  * Con recursos limitados, LoRA o QLoRA son las opciones preferidas.

* Dominio de aplicación:
  * Para dominios muy especializados, SFT con datos de alta calidad puede ser suficiente.
  * Para sistemas interactivos (asistentes, chatbots), RLHF mejora la experiencia de usuario.

* Pipeline de despliegue:
  * Usar librerías consolidadas (por ejemplo, Hugging Face PEFT) para LoRA/QLoRA.
  * Automatizar la recolección de datos de preferencia y el entrenamiento de Reward Models para RLHF, considerando DPO o RLAIF para escalar.

* Mantenimiento y actualización:
  * Guardar adaptadores LoRA por separado para cambiar rápidamente de tarea.
  * Monitorizar la calidad tras el despliegue y programar actualizaciones periódicas con nuevos datos de feedback.

Así, la elección entre estas técnicas depende de factores como el presupuesto de GPU, la disponibilidad de datos de calidad, la necesidad de cambiar rápidamente de tarea y el grado de alineamiento humano que se persiga. En la práctica, a menudo se combinan varias: por ejemplo, emplear LoRA o adapters para las etapas más frecuentes de ajuste rápido, y reservar un pequeño ciclo de PPO o DPO para pulir el comportamiento en producción con datos reales de usuarios.
