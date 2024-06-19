Repaso
------

**Redes Generativas Adversarias**

A continuación se presenta el resumen del artículo original sobre las
Generative Adversarial Networks. Al leer este resumen, notarás muchos
términos y conceptos con los que quizás no estés familiarizado.

Fuente: https://arxiv.org/abs/1406.2661

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David
Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

Proponemos un nuevo marco para estimar modelos generativos a través de
un proceso adversarial, en el cual entrenamos simultáneamente dos
modelos: un modelo generativo G que captura la distribución de los
datos, y un modelo discriminativo D que estima la probabilidad de que
una muestra provenga de los datos de entrenamiento en lugar de G. El
procedimiento de entrenamiento para G es maximizar la probabilidad de
que D cometa un error. Este marco corresponde a un juego minimax de dos
jugadores. En el espacio de funciones arbitrarias G y D, existe una
solución única, con G recuperando la distribución de los datos de
entrenamiento y D igual a 1/2 en todas partes. En el caso donde G y D
están definidos por perceptrones multicapa, todo el sistema puede ser
entrenado con retropropagación. No se necesita ninguna cadena de Markov
ni redes de inferencia aproximada desenvueltas durante el entrenamiento
o la generación de muestras. Los experimentos demuestran el potencial
del marco a través de una evaluación cualitativa y cuantitativa de las
muestras generadas.

**Attention Is All You Need**

Fuente: https://arxiv.org/abs/1706.03762

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

Los modelos de transducción de secuencias dominantes se basan en redes
neuronales recurrentes o convolucionales complejas en una configuración
de codificador-decodificador. Los modelos de mejor rendimiento también
conectan el codificador y el decodificador a través de un mecanismo de
atención. Proponemos una nueva arquitectura de red simple, el
Transformer, basada únicamente en mecanismos de atención, eliminando por
completo la recurrencia y las convoluciones. Experimentos en dos tareas
de traducción automática muestran que estos modelos son superiores en
calidad, al tiempo que son más paralelizables y requieren
significativamente menos tiempo de entrenamiento. Nuestro modelo logra
28.4 BLEU en la tarea de traducción de inglés a alemán del WMT 2014,
mejorando sobre los mejores resultados existentes, incluidas las
combinaciones, en más de 2 BLEU. En la tarea de traducción de inglés a
francés del WMT 2014, nuestro modelo establece una nueva puntuación BLEU
de estado del arte para un solo modelo de 41.8 después de entrenar
durante 3.5 días en ocho GPUs, una fracción pequeña de los costos de
entrenamiento de los mejores modelos de la literatura. Demostramos que
el Transformer se generaliza bien a otras tareas aplicándolo con éxito
al análisis sintáctico del inglés tanto con datos de entrenamiento
grandes como limitados.

**GPT-4 Technical Report (Abstract)**

Fuente: https://arxiv.org/abs/2303.08774

Informamos sobre el desarrollo de GPT-4, un modelo multimodal a gran
escala que puede aceptar entradas de texto e imagen y producir salidas
de texto. Aunque es menos capaz que los humanos en muchos escenarios del
mundo real, GPT-4 exhibe un rendimiento a nivel humano en varios puntos
de referencia profesionales y académicos, incluyendo aprobar un examen
simulado de abogacía con una puntuación en el 10% superior de los
examinados. GPT-4 es un modelo basado en Transformer preentrenado para
predecir el siguiente token en un documento. El proceso de alineación
postentrenamiento resulta en un mejor rendimiento en medidas de
factualidad y adherencia al comportamiento deseado. Un componente
central de este proyecto fue desarrollar infraestructura y métodos de
optimización que se comporten de manera predecible en una amplia gama de
escalas. Esto nos permitió predecir con precisión algunos aspectos del
rendimiento de GPT-4 basándonos en modelos entrenados con no más de
1/1,000 del cómputo de GPT-4.

**Training Language Models to Follow Instructions with Human Feedback**

Fuente: https://arxiv.org/abs/2203.02155

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright,
Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray,
John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens,
Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe

Hacer que los modelos de lenguaje sean más grandes no los hace
inherentemente mejores para seguir la intención del usuario. Por
ejemplo, los modelos de lenguaje grandes pueden generar salidas que son
falsas, tóxicas o simplemente no útiles para el usuario. En otras
palabras, estos modelos no están alineados con sus usuarios. En este
artículo, mostramos una vía para alinear los modelos de lenguaje con la
intención del usuario en una amplia gama de tareas afinándolos con
retroalimentación humana. Comenzando con un conjunto de indicaciones
escritas por etiquetadores e indicaciones enviadas a través de la API de
OpenAI, recopilamos un conjunto de datos de demostraciones de
etiquetadores del comportamiento deseado del modelo, que usamos para
afinar GPT-3 usando aprendizaje supervisado. Luego recopilamos un
conjunto de datos de clasificaciones de salidas del modelo, que usamos
para afinar aún más este modelo supervisado utilizando aprendizaje por
refuerzo con retroalimentación humana. Llamamos a los modelos
resultantes InstructGPT. En evaluaciones humanas de nuestra distribución
de indicaciones, las salidas del modelo InstructGPT de 1.3B parámetros
son preferidas a las salidas del GPT-3 de 175B parámetros, a pesar de
tener 100 veces menos parámetros. Además, los modelos InstructGPT
muestran mejoras en veracidad y reducciones en la generación de salidas
tóxicas, mientras tienen regresiones mínimas de rendimiento en conjuntos
de datos públicos de NLP. Aunque InstructGPT aún comete errores simples,
nuestros resultados muestran que afinar con retroalimentación humana es
una dirección prometedora para alinear los modelos de lenguaje con la
intención humana.

**Denoising Diffusion Probabilistic Models**

Fuente: https://arxiv.org/abs/2006.11239

Jonathan Ho, Ajay Jain, Pieter Abbeel

Presentamos resultados de síntesis de imágenes de alta calidad
utilizando modelos probabilísticos de difusión, una clase de modelos de
variables latentes inspirados en consideraciones de la termodinámica
fuera del equilibrio. Nuestros mejores resultados se obtienen entrenando
con una cota variacional ponderada diseñada según una nueva conexión
entre modelos probabilísticos de difusión y coincidencia de puntuación
de desenfoque con dinámica de Langevin, y nuestros modelos admiten
naturalmente un esquema de descompresión con pérdida progresiva que
puede interpretarse como una generalización de la decodificación
autoregresiva. En el conjunto de datos incondicional CIFAR10, obtenemos
una puntuación de Inception de 9.46 y una puntuación FID de 3.17, ambas
de vanguardia. En LSUN de 256x256, obtenemos una calidad de muestra
similar a ProgressiveGAN.

Hugging Face orientado a LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1 . Instalación y configuración

Para comenzar a trabajar con la biblioteca Hugging Face Transformers,
primero debes instalarla junto con sus dependencias. Utiliza pip para
instalarlo

.. code:: ipython3

    !pip install transformers datasets torch optuna


2 . Carga y uso de modelos preentrenados

Hugging Face ofrece una amplia variedad de modelos preentrenados que
puedes cargar y usar fácilmente.

.. code:: ipython3

    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Cargar el tokenizador y el modelo
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Entrada de texto
    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generar texto con el pad_token_id y attention_mask
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_length=50, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    


.. code:: ipython3

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # Cargar el modelo y el tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Tokenizar una oración
    inputs = tokenizer("Este es un ejemplo.", return_tensors="pt")
    
    # Realizar la predicción
    outputs = model(**inputs)


3 . Fine-Tuning de modelos

Fine-tuning es el proceso de ajustar un modelo preentrenado en un
conjunto de datos específico para mejorar su desempeño en una tarea
concreta. A continuación se muestra un ejemplo básico de cómo realizar
fine-tuning en un conjunto de datos personalizado.

**Prepara el conjunto de datos**

Para este ejemplo, usaremos un conjunto de datos de Hugging Face
Datasets.

.. code:: ipython3

    from datasets import load_dataset
    
    # Cargar el conjunto de datos
    dataset = load_dataset("yelp_review_full")
    


**Tokenización del conjunto de datos** Es necesario tokenizar los datos
para que el modelo los entienda.

.. code:: ipython3

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)


**Dividir el conjunto de datos**

Dividimos el conjunto de datos en conjuntos de entrenamiento y
validación.

.. code:: ipython3

    # Dividir en entrenamiento y evaluación
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


**Preparar el modelo para Fine-Tuning**

Cargamos el modelo preentrenado y lo preparamos para la tarea de
clasificación de texto.

.. code:: ipython3

    from transformers import TrainingArguments, Trainer
    
    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        optim="adamw_torch"  # Usar la implementación de AdamW de PyTorch
    )
    


**Configurar el entrenamiento**

Configuramos los parámetros de entrenamiento y usamos Trainer para
entrenar el modelo.

.. code:: ipython3

    # Crear el entrenador
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Entrenar el modelo
    trainer.train()

4 . Evaluación del modelo

Después del entrenamiento, evaluamos el modelo para ver cómo se
desempeña en el conjunto de datos de prueba.

.. code:: ipython3

    # Evaluar el modelo
    eval_results = trainer.evaluate()
    print(eval_results)


5 . Uso del modelo Fine-Tuned

Finalmente, podemos usar el modelo entrenado para hacer predicciones en
nuevos datos.

.. code:: ipython3

    # Usar el modelo fine-tuned para realizar predicciones
    inputs = tokenizer("Este es un ejemplo.", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)

6 . Guardado y carga del modelo

Es importante guardar el modelo entrenado para su uso futuro. Aquí te
mostramos cómo guardar y cargar el modelo.

**Guardar el modelo**

.. code:: ipython3

    # Guardar el modelo y el tokenizer
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

**Cargar el modelo**

.. code:: ipython3

    # Cargar el modelo y el tokenizer guardados
    model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

7 . Optimización y mejoras

**Ajuste de hiperparámetros**

El ajuste de hiperparámetros puede mejorar significativamente el
rendimiento del modelo. Esto implica experimentar con diferentes valores
de hiperparámetros como la tasa de aprendizaje, el tamaño del batch,
etc.

**Uso de Optuna para optimización de hiperparámetros**

`Optuna <https://optuna.org/>`__ es una biblioteca para la optimización
automática de hiperparámetros.

.. code:: ipython3

    import optuna
    
    def objective(trial):
        # Definir los hiperparámetros que se desean ajustar
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)
        
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        return eval_results['eval_loss']
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    print(study.best_params)


8 . Implementación del modelo en producción

**Usando Hugging Face Inference API**

Puedes usar la API de Hugging Face para implementar modelos en
producción de manera sencilla.

**Despliegue en Amazon SageMaker**

Hugging Face también ofrece integración con Amazon SageMaker para
despliegue escalable.

.. code:: ipython3

    from sagemaker.huggingface import HuggingFaceModel
    
    # Definir el modelo Hugging Face
    huggingface_model = HuggingFaceModel(
       model_data="s3://path/to/your/model.tar.gz",   # path to your trained model
       role=role,                                     # IAM role with SageMaker permissions
       transformers_version="4.6",
       pytorch_version="1.7",
       py_version="py36",
    )
    
    # Despliegue del modelo
    predictor = huggingface_model.deploy(
       initial_instance_count=1,
       instance_type="ml.m5.xlarge"
    )


9 . Trabajando con modelos de traducción

Hugging Face también ofrece modelos para traducción automática.

A continuación se muestra cómo usar un modelo de traducción
preentrenado.

.. code:: ipython3

    from transformers import MarianMTModel, MarianTokenizer
    
    # Cargar el modelo y el tokenizer para traducción
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Realizar traducción
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(translated_text)


10 . Uso de modelos de resumen de texto

.. code:: ipython3

    from transformers import BartForConditionalGeneration, BartTokenizer
    
    # Cargar el modelo y el tokenizer para resumen
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Realizar resumen
    text = "Tu texto largo aquí."
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(summary)


11 . Ajuste de parámetros de generación

.. code:: ipython3

    # Ajustar parámetros de generación como num_beams, length_penalty, etc.
    generated_ids = model.generate(inputs["input_ids"], num_beams=5, length_penalty=1.5, max_length=150, min_length=50)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)


12 . Evaluación automática de modelos

Para evaluar automáticamente los modelos, puedes utilizar métricas como
BLEU, ROUGE, etc.

.. code:: ipython3

    from datasets import load_metric
    
    metric = load_metric("rouge")
    predictions = ["Tu texto generado"]
    references = ["Tu texto de referencia"]
    results = metric.compute(predictions=predictions, references=references)
    print(results)


13 . Integración con aplicaciones Web

Puedes integrar los modelos con aplicaciones web utilizando frameworks
como Flask o FastAPI.

.. code:: ipython3

    from flask import Flask, request, jsonify
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        inputs = tokenizer(data['text'], return_tensors="pt")
        outputs = model.generate(**inputs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'generated_text': generated_text})
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)


14 . Entrenamiento distribuido y acelerado

Para el entrenamiento distribuido y acelerado, puedes utilizar el
soporte de Hugging Face para aceleradores de hardware como GPUs y TPUs.

.. code:: ipython3

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        # Añadir soporte para múltiples GPUs
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,  # Usar precisión de 16 bits
        deepspeed="path/to/deepspeed_config.json"  # Configuración de DeepSpeed para entrenamiento distribuido
    )
    
    # Configurar y entrenar el modelo de manera distribuida
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()


Ejemplos
~~~~~~~~

1 . Fine-Tuning y evaluación de un Modelo BERT para Clasificación de
Texto

Este ejemplo abarca la carga de un conjunto de datos, el
preprocesamiento, el fine-tuning de un modelo BERT para clasificación de
texto, y la evaluación del modelo utilizando métricas avanzadas.

.. code:: ipython3

    import torch
    from datasets import load_dataset, load_metric
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    
    # Cargar el conjunto de datos y el tokenizador
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Preprocesar el conjunto de datos
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))  # Usamos un subconjunto para el ejemplo
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    # Cargar el modelo
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    # Definir una función de evaluación
    def compute_metrics(p):
        metric = load_metric("accuracy")
        predictions, labels = p
        predictions = predictions.argmax(axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Crear el objeto Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # Evaluar el modelo
    results = trainer.evaluate()
    print("Evaluation results:", results)
    
    # Guardar el modelo y el tokenizador
    model.save_pretrained("./fine_tuned_bert")
    tokenizer.save_pretrained("./fine_tuned_bert")
    
    # Ejemplo de uso del modelo fine-tuned
    def classify_text(text):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "positive" if prediction == 1 else "negative"
    
    sample_text = "This movie was absolutely fantastic!"
    print(f"Sample text classification: {classify_text(sample_text)}")


2 . Implementación de un modelo de traducción con evaluación BLEU

Este ejemplo muestra cómo cargar un modelo de traducción, realizar
traducción automática y evaluar el rendimiento del modelo utilizando la
métrica BLEU.

.. code:: ipython3

    from transformers import MarianMTModel, MarianTokenizer
    from datasets import load_dataset, load_metric
    import numpy as np
    
    # Cargar el modelo y el tokenizador
    model_name = 'Helsinki-NLP/opus-mt-en-de'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Cargar el conjunto de datos WMT14
    dataset = load_dataset('wmt14', 'de-en', split='test[:1%]')  # Usamos una muestra pequeña para el ejemplo
    
    # Tokenizar el texto de origen
    def preprocess_function(examples):
        inputs = tokenizer(examples['en'], truncation=True, padding='longest', return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['de'], truncation=True, padding='longest', return_tensors="pt")
        inputs['labels'] = labels['input_ids']
        return inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
    train_dataset = tokenized_datasets.shuffle(seed=42).select(range(500))  # Usamos un subconjunto para el ejemplo
    
    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    # Definir una función de evaluación BLEU
    def compute_metrics(p):
        metric = load_metric("sacrebleu")
        predictions = p.predictions
        labels = p.label_ids
    
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        decoded_labels = [[label] for label in decoded_labels]
    
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
    
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
    
        return result
    
    # Crear el objeto Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Para este ejemplo, usamos el mismo conjunto para evaluación
        compute_metrics=compute_metrics,
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # Evaluar el modelo
    results = trainer.evaluate()
    print("BLEU evaluation results:", results)
    
    # Guardar el modelo y el tokenizador
    model.save_pretrained("./fine_tuned_translation_model")
    tokenizer.save_pretrained("./fine_tuned_translation_model")
    
    # Ejemplo de traducción
    def translate_text(text):
        inputs = tokenizer(text, return_tensors="pt")
        translated_tokens = model.generate(**inputs)
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translation
    
    sample_text = "This is a test sentence for translation."
    print(f"Sample translation: {translate_text(sample_text)}")


3 . Entrenamiento y evaluación de un modelo de resumen de texto con
métricas ROUGE

Este ejemplo muestra cómo entrenar un modelo de resumen de texto y
evaluar su rendimiento utilizando métricas ROUGE.

.. code:: ipython3

    from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
    from datasets import load_dataset, load_metric
    import numpy as np
    
    # Cargar el modelo y el tokenizador
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Cargar el conjunto de datos CNN/DailyMail
    dataset = load_dataset("cnn_dailymail", "3.0.0", split='train[:1%]')  # Usamos una muestra pequeña para el ejemplo
    
    # Preprocesar el conjunto de datos
    def preprocess_function(examples):
        inputs = tokenizer(examples['article'], truncation=True, padding='longest', return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['highlights'], truncation=True, padding='longest', return_tensors="pt")
        inputs['labels'] = labels['input_ids']
        return inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights"])
    train_dataset = tokenized_datasets.shuffle(seed=42).select(range(500))  # Usamos un subconjunto para el ejemplo
    
    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    # Definir una función de evaluación ROUGE
    def compute_metrics(p):
        metric = load_metric("rouge")
        predictions = p.predictions
        labels = p.label_ids
    
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
    
        return result
    
    # Crear el objeto Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Para este ejemplo, usamos el mismo conjunto para evaluación
        compute_metrics=compute_metrics,
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # Evaluar el modelo
    results = trainer.evaluate()
    print("ROUGE evaluation results:", results)
    
    # Guardar el modelo y el tokenizador
    model.save_pretrained("./fine_tuned_summarization_model")
    tokenizer.save_pretrained("./fine_tuned_summarization_model")
    
    # Ejemplo de resumen
    def summarize_text(text):
        inputs = tokenizer(text, return_tensors="pt")
        summary_ids = model.generate(inputs['input_ids'], max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    sample_text = "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy."
    print(f"Sample summary: {summarize_text(sample_text)}")


4 . Implementación de un pipeline de clasificación de texto con FastAPI

Este ejemplo muestra cómo entrenar un modelo de clasificación de texto y
luego desplegarlo en un servicio web utilizando FastAPI.

.. code:: ipython3

    # Importar bibliotecas necesarias
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    from datasets import load_dataset, load_metric
    
    # Crear la aplicación FastAPI
    app = FastAPI()
    
    # Definir la entrada del modelo
    class TextItem(BaseModel):
        text: str
    
    # Cargar el modelo y el tokenizador entrenados
    model_path = "./fine_tuned_bert"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Crear una función de predicción
    def predict(text: str):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "positive" if prediction == 1 else "negative"
    
    # Crear un endpoint de predicción
    @app.post("/predict/")
    def classify_text(item: TextItem):
        try:
            prediction = predict(item.text)
            return {"text": item.text, "classification": prediction}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Para correr el servidor, usa:
    # uvicorn script_name:app --reload


5 . Traducción automática con evaluación de BLEU y despliegue en Flask

Este ejemplo muestra cómo entrenar un modelo de traducción automática,
evaluar su rendimiento utilizando la métrica BLEU y desplegarlo en un
servicio web utilizando Flask.

.. code:: ipython3

    from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
    from datasets import load_dataset, load_metric
    from flask import Flask, request, jsonify
    import numpy as np
    
    # Cargar el modelo y el tokenizador
    model_name = 'Helsinki-NLP/opus-mt-en-de'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Cargar el conjunto de datos WMT14
    dataset = load_dataset('wmt14', 'de-en', split='test[:1%]')
    
    # Preprocesar el conjunto de datos
    def preprocess_function(examples):
        inputs = tokenizer(examples['en'], truncation=True, padding='longest', return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['de'], truncation=True, padding='longest', return_tensors="pt")
        inputs['labels'] = labels['input_ids']
        return inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
    train_dataset = tokenized_datasets.shuffle(seed=42).select(range(500))
    
    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    # Definir una función de evaluación BLEU
    def compute_metrics(p):
        metric = load_metric("sacrebleu")
        predictions = p.predictions
        labels = p.label_ids
    
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        decoded_labels = [[label] for label in decoded_labels]
    
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
    
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
    
        return result
    
    # Crear el objeto Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # Evaluar el modelo
    results = trainer.evaluate()
    print("BLEU evaluation results:", results)
    
    # Guardar el modelo y el tokenizador
    model.save_pretrained("./fine_tuned_translation_model")
    tokenizer.save_pretrained("./fine_tuned_translation_model")
    
    # Crear la aplicación Flask
    app = Flask(__name__)
    
    # Definir la entrada del modelo
    class TranslationItem(BaseModel):
        text: str
    
    # Crear una función de traducción
    def translate(text: str):
        inputs = tokenizer(text, return_tensors="pt")
        translated_tokens = model.generate(**inputs)
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translation
    
    # Crear un endpoint de traducción
    @app.route("/translate", methods=["POST"])
    def translate_text():
        try:
            data = request.get_json()
            translation = translate(data['text'])
            return jsonify({"text": data['text'], "translation": translation})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Para correr el servidor, usa:
    # flask run


6 . Resumen de texto con evaluación ROUGE y despliegue en FastAPI

Este ejemplo muestra cómo entrenar un modelo de resumen de texto,
evaluar su rendimiento utilizando métricas ROUGE y desplegarlo en un
servicio web utilizando FastAPI.

.. code:: ipython3

    from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
    from datasets import load_dataset, load_metric
    import numpy as np
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    # Cargar el modelo y el tokenizador
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Cargar el conjunto de datos CNN/DailyMail
    dataset = load_dataset("cnn_dailymail", "3.0.0", split='train[:1%]')
    
    # Preprocesar el conjunto de datos
    def preprocess_function(examples):
        inputs = tokenizer(examples['article'], truncation=True, padding='longest', return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['highlights'], truncation=True, padding='longest', return_tensors="pt")
        inputs['labels'] = labels['input_ids']
        return inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights"])
    train_dataset = tokenized_datasets.shuffle(seed=42).select(range(500))
    
    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    # Definir una función de evaluación ROUGE
    def compute_metrics(p):
        metric = load_metric("rouge")
        predictions = p.predictions
        labels = p.label_ids
    
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
    
        return result
    
    # Crear el objeto Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # Evaluar el modelo
    results = trainer.evaluate()
    print("ROUGE evaluation results:", results)
    
    # Guardar el modelo y el tokenizador
    model.save_pretrained("./fine_tuned_summarization_model")
    tokenizer.save_pretrained("./fine_tuned_summarization_model")
    
    # Crear la aplicación FastAPI
    app = FastAPI()
    
    # Definir la entrada del modelo
    class SummaryItem(BaseModel):
        text: str
    
    # Crear una función de resumen
    def summarize(text: str):
        inputs = tokenizer(text, return_tensors="pt")
        summary_ids = model.generate(inputs['input_ids'], max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    # Crear un endpoint de resumen
    @app.post("/summarize/")
    def summarize_text(item: SummaryItem):
        try:
            summary = summarize(item.text)
            return {"text": item.text, "summary": summary}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Para correr el servidor, usa:
    # uvicorn script_name:app --reload


7 . Entrenamiento distribuido con PyTorch Lightning y DeepSpeed

Este ejemplo muestra cómo utilizar PyTorch Lightning y DeepSpeed para
entrenar modelos de manera distribuida y eficiente.

.. code:: ipython3

    import torch
    import pytorch_lightning as pl
    from transformers import BertForSequenceClassification, BertTokenizer
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import deepspeed
    
    # Cargar el conjunto de datos y el tokenizador
    dataset = load_dataset("imdb")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Preprocesar el conjunto de datos
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    # Crear un DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16)
    
    # Definir el modelo de clasificación con BERT
    class BertClassifier(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
            self.criterion = torch.nn.CrossEntropyLoss()
    
        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            return outputs
    
        def training_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = outputs.loss
            self.log("train_loss", loss)
            return loss
    
        def validation_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = outputs.loss
            self.log("val_loss", loss)
            return loss
    
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=2e-5)
    
    # Configurar el entrenador de PyTorch Lightning con DeepSpeed
    trainer = pl.Trainer(
        gpus=1, 
        precision=16,  # Para FP16
        plugins=[deepspeed.DeepSpeedPlugin(config={"train_batch_size": 16})],
        max_epochs=3
    )
    
    # Entrenar el modelo
    model = BertClassifier()
    trainer.fit(model, train_dataloader, eval_dataloader)
    
    # Guardar el modelo y el tokenizador
    model.model.save_pretrained("./deepspeed_bert")
    tokenizer.save_pretrained("./deepspeed_bert")


8 . Retrieval-Augmented Generation (RAG)

RAG combina la generación de texto con la recuperación de documentos
relevantes. Aquí se muestra cómo configurar y usar un modelo RAG.

.. code:: ipython3

    from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
    
    # Cargar el tokenizador y el modelo
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
    
    # Texto de entrada
    input_text = "What is the capital of France?"
    
    # Tokenizar y generar
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    generated = model.generate(input_ids, num_beams=2, num_return_sequences=2)
    
    # Decodificar la salida
    generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    print(generated_texts)


9 . GPT-3 en modo Zero-Shot

En el modo Zero-Shot, utilizamos el modelo GPT-3 sin entrenamiento
adicional para realizar tareas como la clasificación o generación de
texto.

.. code:: ipython3

    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    # Cargar el tokenizador y el modelo
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Texto de entrada para Zero-Shot
    input_text = "Translate English to French: What time is it?"
    
    # Tokenizar y generar
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=40)
    
    # Decodificar la salida
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)


10 . GPT-3 en modo Few-Shot

En el modo Few-Shot, proporcionamos algunos ejemplos de la tarea al
modelo para guiar su generación.

.. code:: ipython3

    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    # Cargar el tokenizador y el modelo
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Texto de entrada para Few-Shot
    input_text = """
    Translate English to French:
    English: What time is it?
    French: Quelle heure est-il?
    
    English: How are you?
    French: Comment ça va?
    
    English: I am fine.
    French:
    """
    
    # Tokenizar y generar
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)
    
    # Decodificar la salida
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)


11 . Implementación completa de un pipeline de RAG

Este ejemplo avanzado muestra cómo implementar un pipeline completo
utilizando RAG, incluyendo la configuración del entorno, recuperación de
documentos y generación de respuestas.

.. code:: ipython3

    from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
    import torch
    
    # Cargar el tokenizador, el recuperador y el modelo
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
    
    # Configuración del entorno
    input_text = "Explain the theory of relativity."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Generar con RAG
    generated = model.generate(input_ids, num_return_sequences=1, num_beams=5)
    generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    print("Generated Texts:", generated_texts)
    
    # Recuperar documentos relevantes
    docs = retriever(input_ids.numpy().flatten())
    retrieved_texts = docs['documents'][0]
    print("Retrieved Texts:", retrieved_texts)
    
    # Generar respuestas basadas en los documentos recuperados
    context = " ".join(retrieved_texts)
    context_input_ids = tokenizer(context, return_tensors="pt").input_ids
    context_generated = model.generate(context_input_ids, num_return_sequences=1, num_beams=5)
    context_generated_texts = [tokenizer.decode(cg, skip_special_tokens=True) for cg in context_generated]
    print("Context-Based Generated Texts:", context_generated_texts)


12 . Comparación de GPT-3 en Modo Zero-Shot y Few-Shot

Este ejemplo muestra cómo comparar el rendimiento del modelo GPT-3 en
modos Zero-Shot y Few-Shot para una tarea específica.

.. code:: ipython3

    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    # Cargar el tokenizador y el modelo
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Modo Zero-Shot
    input_text_zero_shot = "Translate English to French: Where is the library?"
    inputs_zero_shot = tokenizer(input_text_zero_shot, return_tensors="pt")
    outputs_zero_shot = model.generate(inputs_zero_shot.input_ids, max_length=50)
    generated_text_zero_shot = tokenizer.decode(outputs_zero_shot[0], skip_special_tokens=True)
    print("Zero-Shot Translation:", generated_text_zero_shot)
    
    # Modo Few-Shot
    input_text_few_shot = """
    Translate English to French:
    English: Where is the library?
    French: Où est la bibliothèque?
    
    English: How are you?
    French: Comment ça va?
    
    English: What time is it?
    French:
    """
    inputs_few_shot = tokenizer(input_text_few_shot, return_tensors="pt")
    outputs_few_shot = model.generate(inputs_few_shot.input_ids, max_length=100)
    generated_text_few_shot = tokenizer.decode(outputs_few_shot[0], skip_special_tokens=True)
    print("Few-Shot Translation:", generated_text_few_shot)


Ejercicios
~~~~~~~~~~

1 . Fine-Tuning y Evaluación de RoBERTa para Clasificación de Texto

Objetivo: Realizar fine-tuning del modelo RoBERTa en un conjunto de
datos específico y evaluar su rendimiento.

Carga y preprocesamiento del conjunto de Datos:

-  Carga el conjunto de datos ag_news utilizando la biblioteca datasets.
-  Tokeniza el conjunto de datos utilizando RobertaTokenizer.

Configuración y entrenamiento:

-  Configura el modelo RobertaForSequenceClassification para la tarea de
   clasificación de texto.
-  Configura los argumentos de entrenamiento utilizando
   TrainingArguments.
-  Entrena el modelo utilizando Trainer.

Evaluación:

-  Evalúa el modelo en el conjunto de datos de prueba.
-  Calcula y muestra las métricas de precisión y F1-score.

.. code:: ipython3

    ## Tu respuesta

2 . Generación de texto con GPT-3 en Modo Few-Shot

Objetivo: Usar GPT-3 en modo Few-Shot para generar texto basado en
ejemplos proporcionados.

Preparación de Prompt:

-  Define ejemplos de texto para guiar al modelo en la tarea de
   generación.

Generación de texto:

-  Usa la API de OpenAI para generar texto basado en los ejemplos
   proporcionados.

Evaluación de resultados:

-  Compara la calidad del texto generado con los ejemplos
   proporcionados.

.. code:: ipython3

    ## Tu respuesta

3 . : Traducción Automática con XLM y Evaluación BLEU

Objetivo: Utilizar XLM para traducción automática y evaluar el
rendimiento utilizando la métrica BLEU.

Preparación del conjunto de datos:

-  Carga un conjunto de datos bilingüe para la traducción.

Traducción automática:

-  Usa XLMTokenizer y XLMWithLMHeadModel para traducir texto.

Evaluación:

-  Evalua las traducciones generadas utilizando la métrica BLEU.

.. code:: ipython3

    ## Tu respuesta

4 . Resumen de texto con T5 y evaluación ROUGE

Objetivo: Utilizar T5 para resumen de texto y evaluar el rendimiento
utilizando la métrica ROUGE.

Carga y preprocesamiento del conjunto de datos:

-  Carga el conjunto de datos cnn_dailymail utilizando la biblioteca
   datasets.
-  Tokeniza el conjunto de datos utilizando T5Tokenizer.

Generación de resúmenes:

-  Usa T5ForConditionalGeneration para generar resúmenes de texto.

Evaluación:

-  Evalua los resúmenes generados utilizando la métrica ROUGE

.. code:: ipython3

    from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
    from datasets import load_dataset, load_metric
    
    # 1. Carga y Preprocesamiento del Conjunto de Datos
    dataset = load_dataset("cnn_dailymail", "3.0.0", split='train[:1%]')
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    def preprocess_function(examples):
        inputs = tokenizer(examples['article'], truncation=True, padding='longest', return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['highlights'], truncation=True, padding='longest', return_tensors="pt")
        inputs['labels'] = labels['input_ids']
        return inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights"])
    train_dataset = tokenized_datasets.shuffle(seed=42).select(range(500))
    
    # 2. Configuración y Entrenamiento
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        compute_metrics=lambda p: load_metric("rouge").compute(predictions=[tokenizer.decode(g, skip_special_tokens=True) for g in p.predictions], references=[[tokenizer.decode(g, skip_special_tokens=True)] for g in p.label_ids])
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # 3. Evaluación
    results = trainer.evaluate()
    print("ROUGE evaluation results:", results)


.. code:: ipython3

    ## Tu respuesta

Introducción a LLM, LangChain y LLaMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Los modelos de lenguaje de gran escala (LLM) han revolucionado el campo
de la inteligencia artificial y el procesamiento del lenguaje natural
(NLP). Con la capacidad de entender y generar texto de manera coherente
y contextualmente precisa, los LLM están transformando la manera en que
interactuamos con la tecnología. Dentro de este contexto, surgen
herramientas y marcos como LangChain y LLaMA, que ofrecen capacidades
avanzadas para el desarrollo y la implementación de aplicaciones basadas
en LLM.

Los LLM, como GPT-4 de OpenAI, BERT de Google y muchos otros, son redes
neuronales profundas entrenadas en grandes volúmenes de datos textuales.
Estos modelos pueden comprender y generar texto en múltiples idiomas,
realizar tareas de traducción, responder preguntas, resumir textos,
entre otras aplicaciones. La clave de su éxito radica en su capacidad
para captar matices y contextos complejos en el lenguaje humano, gracias
a técnicas de aprendizaje profundo y enormes cantidades de datos de
entrenamiento.

La arquitectura detrás de los LLM suele estar basada en transformers,
una estructura de red neuronal que permite procesar secuencias de datos
de manera más eficiente y efectiva que las arquitecturas anteriores,
como las redes recurrentes. Los transformers utilizan mecanismos de
atención que permiten a los modelos enfocarse en partes relevantes del
texto al procesar información, mejorando así la comprensión y generación
de texto.

LangChain: Integración y expansión de capacidades LLM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LangChain es un marco diseñado para facilitar la creación de
aplicaciones basadas en LLM. Proporciona herramientas y bibliotecas que
permiten a los desarrolladores integrar modelos de lenguaje en sus
aplicaciones de manera más sencilla y efectiva. LangChain ofrece una
serie de funcionalidades clave, incluyendo la gestión de diálogos, la
integración con bases de datos, la ejecución de tareas específicas y la
personalización de respuestas.

Una de las características destacadas de LangChain es su capacidad para
gestionar contextos prolongados y mantener coherencia en interacciones
de varios turnos. Esto es particularmente útil en aplicaciones como
chatbots y asistentes virtuales, donde es crucial mantener el contexto
de la conversación a lo largo del tiempo. LangChain logra esto mediante
el uso de técnicas avanzadas de manejo de estados y contextos,
permitiendo que los LLM proporcionen respuestas coherentes y relevantes.

Además, LangChain facilita la integración de modelos de lenguaje con
otros sistemas y fuentes de datos. Por ejemplo, permite a los
desarrolladores conectar LLM con bases de datos SQL y NoSQL, APIs
externas, y sistemas de gestión de contenido, ampliando
significativamente las capacidades de las aplicaciones basadas en LLM.
Esto abre un abanico de posibilidades para crear soluciones
personalizadas y adaptadas a necesidades específicas de diferentes
industrias.

LLaMA: Modelos de lenguaje de libre acceso
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`LLaMA <https://ai.meta.com/blog/large-language-model-llama-meta-ai/>`__,
que significa “Large Language Model Accessibility”, es un proyecto que
busca democratizar el acceso a modelos de lenguaje avanzados. A
diferencia de otros modelos que pueden tener restricciones de acceso
debido a costos o limitaciones comerciales, LLaMA se centra en
proporcionar modelos de lenguaje potentes que estén disponibles para una
amplia audiencia, incluyendo investigadores, desarrolladores
independientes y pequeñas empresas.

El objetivo de LLaMA es reducir las barreras de entrada para el uso de
LLM, permitiendo que más personas puedan experimentar con estas
tecnologías y desarrollar aplicaciones innovadoras. LLaMA ofrece una
serie de modelos preentrenados que los usuarios pueden descargar y
utilizar en sus proyectos. Estos modelos están optimizados para
funcionar en hardware accesible, lo que facilita su implementación en
entornos con recursos limitados.

LLaMA también se enfoca en proporcionar documentación y recursos
educativos para ayudar a los usuarios a comprender y utilizar los
modelos de lenguaje de manera efectiva. Esto incluye tutoriales,
ejemplos de código, y guías de mejores prácticas, que son esenciales
para maximizar el potencial de los LLM en aplicaciones prácticas.

Integración de LangChain y LLaMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La combinación de LangChain y LLaMA presenta una poderosa herramienta
para el desarrollo de aplicaciones basadas en LLM. Mientras que
LangChain ofrece un marco robusto para la integración y gestión de
modelos de lenguaje, LLaMA proporciona acceso a modelos de lenguaje de
alta calidad y accesibles. Juntos, estos recursos permiten a los
desarrolladores crear aplicaciones sofisticadas que aprovechan al máximo
las capacidades de los LLM.

Una posible aplicación de esta integración podría ser el desarrollo de
un asistente virtual personalizado para una pequeña empresa. Utilizando
LangChain, el desarrollador puede integrar el asistente con el sistema
de gestión de clientes de la empresa, permitiendo que el asistente
acceda a información relevante y proporcione respuestas precisas y
contextualmente apropiadas. Al mismo tiempo, utilizando un modelo LLaMA,
el desarrollador puede asegurar que el asistente funcione de manera
eficiente en hardware accesible, sin necesidad de inversiones
significativas en infraestructura.

Otra aplicación podría ser en el ámbito educativo, donde se pueden crear
tutores virtuales que ayuden a los estudiantes a aprender nuevos
conceptos y resolver dudas. Estos tutores pueden estar integrados con
bases de datos de contenido educativo y proporcionar explicaciones
detalladas y personalizadas basadas en el progreso y necesidades de cada
estudiante.

Desafíos y Futuro de LLM, LangChain y LLaMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pesar de las numerosas ventajas y aplicaciones de los LLM, LangChain y
LLaMA, también existen desafíos significativos que deben abordarse. Uno
de los principales retos es la ética y la responsabilidad en el uso de
modelos de lenguaje. Los LLM tienen el potencial de generar contenido
que puede ser perjudicial o engañoso si no se utilizan correctamente.
Por lo tanto, es crucial desarrollar y seguir directrices éticas y
mecanismos de control para asegurar el uso responsable de estas
tecnologías.

Otro desafío es la escalabilidad y eficiencia de los modelos de
lenguaje. A medida que los LLM se vuelven más grandes y complejos,
también aumentan los requisitos de computación y almacenamiento. Esto
puede ser una barrera para su adopción en entornos con recursos
limitados. Sin embargo, proyectos como LLaMA están trabajando para
mitigar estos problemas proporcionando modelos optimizados y accesibles.

En términos de futuro, se espera que la integración de LLM en
aplicaciones continúe creciendo y evolucionando. Con la mejora continua
en las arquitecturas de modelos y técnicas de entrenamiento, los LLM
serán cada vez más capaces de comprender y generar texto de manera más
precisa y contextualmente relevante. Además, herramientas como LangChain
y LLaMA seguirán facilitando el desarrollo y la implementación de
aplicaciones basadas en LLM, democratizando el acceso a estas poderosas
tecnologías y permitiendo una innovación más amplia y diversificada.

La combinación de LLM, LangChain y LLaMA representa una convergencia de
tecnología y accesibilidad que tiene el potencial de transformar
numerosos campos y aplicaciones. Al reducir las barreras de entrada y
proporcionar herramientas avanzadas para el desarrollo, estas
tecnologías están preparando el camino para una nueva era de
aplicaciones inteligentes y contextualmente conscientes, beneficiando
tanto a desarrolladores como a usuarios finales.

.. code:: ipython3

    ## Tu respuesta
