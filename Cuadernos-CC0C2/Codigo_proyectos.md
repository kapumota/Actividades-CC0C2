### **1. Configuración de la estructura del repositorio**

Primero, crearemos un script en Python que automatice la creación de la estructura del repositorio con las carpetas `Entrega_1`, `Entrega_2` y `Entrega_3`, así como un archivo `README.md` en cada una de ellas.

#### **a. Script para crear la estructura del repositorio**

```python
import os
from datetime import datetime

# Definir los nombres de las carpetas de entrega
entregas = {
    "Entrega_1": "Hasta el 30 de noviembre",
    "Entrega_2": "Hasta el 7 de diciembre",
    "Entrega_3": "Hasta el 14 de diciembre (Final)"
}

# Función para crear las carpetas y archivos README.md
def crear_estructura_repositorio(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Repositorio creado en: {base_path}")
    else:
        print(f"El repositorio ya existe en: {base_path}")
    
    for carpeta, plazo in entregas.items():
        carpeta_path = os.path.join(base_path, carpeta)
        os.makedirs(carpeta_path, exist_ok=True)
        print(f"Carpeta creada: {carpeta_path}")
        
        # Crear README.md con contenido base
        readme_path = os.path.join(carpeta_path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# {carpeta}\n\n")
                f.write(f"**Plazo de entrega:** {plazo}\n\n")
                f.write("## Avances y Mejoras\n\n")
                f.write("Describe aquí los cambios realizados en esta entrega.\n")
            print(f"README.md creado en: {readme_path}")
        else:
            print(f"README.md ya existe en: {readme_path}")

# Ruta base del repositorio (puedes modificarla según tus necesidades)
ruta_repositorio = os.path.join(os.getcwd(), "Proyecto_Curso")

# Ejecutar la función para crear la estructura
crear_estructura_repositorio(ruta_repositorio)
```

#### **b. Instrucciones para ejecutar el script**

1. **Guardar el script:**
   Guarda el script anterior en un archivo llamado, por ejemplo, `configurar_repositorio.py`.

2. **Ejecutar el script:**
   Abre una terminal, navega hasta el directorio donde guardaste el script y ejecuta:

   ```bash
   python configurar_repositorio.py
   ```

   Esto creará una carpeta llamada `Proyecto_Curso` (puedes cambiar el nombre en la variable `ruta_repositorio`) con las tres carpetas de entrega y un `README.md` en cada una.

### **2. Automatización de control de versiones con Git**

Para gestionar los commits automáticamente cada vez que realizas cambios significativos, puedes extender el script anterior para inicializar un repositorio Git y realizar los primeros commits.

#### **a. Extender el script para inicializar Git**

```python
import os
import subprocess

def inicializar_git(base_path):
    # Inicializar el repositorio Git
    subprocess.run(["git", "init"], cwd=base_path)
    print("Repositorio Git inicializado.")
    
    # Crear un archivo .gitignore básico
    gitignore_path = os.path.join(base_path, ".gitignore")
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write("__pycache__/\n*.pyc\n.env\n")
    print(".gitignore creado.")
    
    # Agregar todos los archivos al staging y realizar el primer commit
    subprocess.run(["git", "add", "."], cwd=base_path)
    subprocess.run(["git", "commit", "-m", "Configuración inicial del repositorio"], cwd=base_path)
    print("Primer commit realizado.")

# Ejecutar la función para inicializar Git
inicializar_git(ruta_repositorio)
```

#### **b. Actualizar el script completo**

Combina ambas funciones en un solo script:

```python
import os
import subprocess
from datetime import datetime

# Definir los nombres de las carpetas de entrega
entregas = {
    "Entrega_1": "Hasta el 30 de noviembre",
    "Entrega_2": "Hasta el 7 de diciembre",
    "Entrega_3": "Hasta el 14 de diciembre (Final)"
}

# Función para crear las carpetas y archivos README.md
def crear_estructura_repositorio(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Repositorio creado en: {base_path}")
    else:
        print(f"El repositorio ya existe en: {base_path}")
    
    for carpeta, plazo in entregas.items():
        carpeta_path = os.path.join(base_path, carpeta)
        os.makedirs(carpeta_path, exist_ok=True)
        print(f"Carpeta creada: {carpeta_path}")
        
        # Crear README.md con contenido base
        readme_path = os.path.join(carpeta_path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# {carpeta}\n\n")
                f.write(f"**Plazo de entrega:** {plazo}\n\n")
                f.write("## Avances y Mejoras\n\n")
                f.write("Describe aquí los cambios realizados en esta entrega.\n")
            print(f"README.md creado en: {readme_path}")
        else:
            print(f"README.md ya existe en: {readme_path}")

# Función para inicializar Git
def inicializar_git(base_path):
    # Inicializar el repositorio Git
    subprocess.run(["git", "init"], cwd=base_path)
    print("Repositorio Git inicializado.")
    
    # Crear un archivo .gitignore básico
    gitignore_path = os.path.join(base_path, ".gitignore")
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write("__pycache__/\n*.pyc\n.env\n")
    print(".gitignore creado.")
    
    # Agregar todos los archivos al staging y realizar el primer commit
    subprocess.run(["git", "add", "."], cwd=base_path)
    subprocess.run(["git", "commit", "-m", "Configuración inicial del repositorio"], cwd=base_path)
    print("Primer commit realizado.")

# Ruta base del repositorio (puedes modificarla según tus necesidades)
ruta_repositorio = os.path.join(os.getcwd(), "Proyecto_Curso")

# Ejecutar las funciones
crear_estructura_repositorio(ruta_repositorio)
inicializar_git(ruta_repositorio)
```

### **3. Ejemplos de inicio para algunos proyectos**

A continuación, proporcionaré ejemplos básicos de cómo podrías comenzar con algunos de los proyectos mencionados, utilizando Python.

#### **Proyecto: Ajustar finamente un LLM con PPO vs DPO vs ORPO utilizando el paquete PEFT**

Este proyecto implica el uso de modelos de lenguaje y técnicas de optimización. A continuación, un ejemplo de cómo podrías comenzar a ajustar finamente un modelo usando el paquete `transformers` de Hugging Face y `peft`.

```python
# Instalar las librerías necesarias
# !pip install transformers peft torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PEFTConfig, get_peft_model, LoraConfig, TaskType
import torch

# Cargar el tokenizer y el modelo base
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Configuración de PEFT (usando LoRA como ejemplo)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# Aplicar PEFT al modelo
model = get_peft_model(model, peft_config)

# Preparar datos de entrenamiento (ejemplo sencillo)
texts = [
    "El clima hoy es soleado y agradable.",
    "La inteligencia artificial está revolucionando el mundo."
]

inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

# Definir optimizador
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Entrenamiento básico
model.train()
for epoch in range(3):
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1} - Loss: {loss.item()}")

# Guardar el modelo ajustado
model.save_pretrained('./modelo_peft_gpt2')
tokenizer.save_pretrained('./modelo_peft_gpt2')
```

#### **Proyecto: Crear una aplicación interactiva de chat que utiliza GPT para responder en tiempo real, con soporte para WebSockets para comunicación continua**

Para este proyecto, podrías usar `FastAPI` para el backend y `Socket.IO` para la comunicación en tiempo real. A continuación, un ejemplo básico del backend.

```python
# Instalar las librerías necesarias
# !pip install fastapi uvicorn python-socketio transformers

from fastapi import FastAPI
import socketio
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Inicializar el servidor SocketIO
sio = socketio.AsyncServer(async_mode='asgi')
app = FastAPI()
sio_app = socketio.ASGIApp(sio, app)

# Cargar el modelo y tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Evento de conexión
@sio.event
async def connect(sid, environ):
    print(f"Usuario conectado: {sid}")

# Evento de desconexión
@sio.event
async def disconnect(sid):
    print(f"Usuario desconectado: {sid}")

# Evento de mensaje
@sio.event
async def message(sid, data):
    print(f"Mensaje recibido de {sid}: {data}")
    # Generar respuesta con GPT-2
    inputs = tokenizer.encode(data, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Enviar respuesta al cliente
    await sio.emit('response', respuesta, to=sid)

# Ejecutar el servidor con Uvicorn
# Ejecuta en la terminal:
# uvicorn nombre_del_script:app --reload
```

#### **Proyecto: Desplegar una API de preguntas y respuestas basada en LLM finamente ajustado con búsqueda de documentos**

Puedes utilizar `FastAPI` para crear una API REST que maneje preguntas y respuestas. A continuación, un ejemplo básico:

```python
# Instalar las librerías necesarias
# !pip install fastapi uvicorn transformers faiss-cpu

from fastapi import FastAPI
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import faiss
import numpy as np

app = FastAPI()

# Cargar modelo y tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Simular una base de datos de documentos
documentos = [
    "La inteligencia artificial está transformando múltiples industrias.",
    "El cambio climático es uno de los mayores desafíos de nuestro tiempo."
]

# Generar embeddings simples (ejemplo con TF-IDF o similar)
# Aquí usamos embeddings aleatorios para simplificar
embeddings = np.random.rand(len(documentos), 768).astype('float32')
index = faiss.IndexFlatL2(768)
index.add(embeddings)

@app.post("/preguntar/")
def responder_pregunta(pregunta: str):
    # Convertir pregunta a embedding (aquí simplificamos)
    pregunta_embedding = np.random.rand(1, 768).astype('float32')
    
    # Buscar documento relevante
    D, I = index.search(pregunta_embedding, 1)
    doc_relevante = documentos[I[0][0]]
    
    # Preparar inputs para el modelo
    inputs = tokenizer.encode_plus(pregunta, doc_relevante, return_tensors='pt')
    input_ids = inputs["input_ids"].tolist()[0]
    
    # Obtener respuestas
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    respuesta = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    return {"respuesta": respuesta, "documento_relevante": doc_relevante}

# Ejecutar el servidor con Uvicorn
# Ejecuta en la terminal:
# uvicorn nombre_del_script:app --reload
```

¡Por supuesto! Continuaré proporcionando soluciones basadas en Python para los proyectos restantes. Cada sección incluirá una descripción inicial, ejemplos de código para comenzar, y referencias a las librerías y herramientas necesarias.

---

### **Proyecto: Entrenar y ajustar un LLM especializado en la clasificación de noticias por temas, usando técnicas de fine-tuning y transfer learning**

#### **a. Configuración inicial**

Para este proyecto, utilizaremos la biblioteca `transformers` de Hugging Face y `datasets` para manejar los datos. Además, emplearemos `scikit-learn` para la evaluación.

#### **b. Instalación de dependencias**

```bash
pip install transformers datasets scikit-learn torch
```

#### **c. Ejemplo de implementación**

```python
from datasets import load_dataset
from transformers import DistilBERTTokenizer, DistilBERTForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Cargar el conjunto de datos
# Aquí puedes usar un conjunto de datos personalizado o uno disponible como "ag_news"
dataset = load_dataset('ag_news')

# Cargar el tokenizer y el modelo preentrenado
tokenizer = DistilBERTTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBERTForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

# Tokenizar los datos
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Definir las métricas de evaluación
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Inicializar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
trainer.evaluate()
```

#### **d. Guardar el modelo ajustado**

```python
model.save_pretrained('./modelo_distilbert_ag_news')
tokenizer.save_pretrained('./modelo_distilbert_ag_news')
```

#### **e. Consideraciones adicionales**

- **Balance de clases:** Si el conjunto de datos está desbalanceado, considera usar técnicas como sobremuestreo (`oversampling`) o asignar pesos a las clases.
- **Optimización de hiperparámetros:** Experimenta con diferentes tasas de aprendizaje, tamaños de batch y número de épocas para optimizar el rendimiento.
- **Interpretabilidad:** Utiliza herramientas como `SHAP` o `LIME` para interpretar las predicciones del modelo.

---

### **Proyecto: Usar embeddings contextuales para generar recomendaciones de películas basadas en descripciones de trama y preferencias del usuario**

#### **a. Configuración inicial**

Utilizaremos `Sentence-BERT` para generar embeddings y `FAISS` para realizar búsquedas eficientes.

#### **b. Instalación de dependencias**

```bash
pip install sentence-transformers faiss-cpu pandas flask
```

#### **c. Ejemplo de implementación**

```python
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

# Cargar el conjunto de datos de películas
# Supongamos que tienes un CSV con una columna 'description'
df = pd.read_csv('movies.csv')  # Asegúrate de tener una columna 'description'

# Generar embeddings de las descripciones
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['description'].tolist(), convert_to_tensor=False)

# Convertir a float32 para FAISS
embeddings = np.array(embeddings).astype('float32')

# Crear el índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Guardar el índice y el dataframe
faiss.write_index(index, 'movies.index')
df.to_csv('movies.csv', index=False)
```

#### **d. Crear una API para recomendaciones**

```python
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el índice
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('movies.index')
df = pd.read_csv('movies.csv')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_preferences = data.get('preferences', '')
    
    # Generar embedding para las preferencias del usuario
    user_embedding = model.encode([user_preferences]).astype('float32')
    
    # Buscar las películas más similares
    D, I = index.search(user_embedding, k=5)  # Top 5 recomendaciones
    
    recommendations = df.iloc[I[0]].to_dict(orient='records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
```

#### **e. Prueba de la API**

Envía una solicitud POST a `http://localhost:5000/recommend` con un JSON como:

```json
{
    "preferences": "Me gustan las películas de ciencia ficción con tramas complejas y efectos especiales impresionantes."
}
```

#### **f. Consideraciones adicionales**

- **Optimización de búsqueda:** Para conjuntos de datos muy grandes, considera usar índices más avanzados de FAISS como `IVF` o `HNSW`.
- **Personalización avanzada:** Incorpora información adicional del usuario, como historial de visualización o calificaciones, para mejorar las recomendaciones.
- **Interfaz de usuario:** Desarrolla un frontend utilizando frameworks como React o Vue.js para una experiencia de usuario más completa.

---

### **Proyecto: Desplegar una API de preguntas y respuestas basada en LLM finamente ajustado con búsqueda de documentos**

#### **a. Configuración inicial**

Emplearemos `FastAPI` para crear la API, `transformers` para el modelo de preguntas y respuestas, y `FAISS` para la búsqueda de documentos.

#### **b. Instalación de dependencias**

```bash
pip install fastapi uvicorn transformers faiss-cpu pandas
```

#### **c. Ejemplo de implementación**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import faiss
import pandas as pd
import numpy as np

app = FastAPI()

# Definir el modelo y el tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Cargar documentos
df = pd.read_csv('documents.csv')  # Supón que tienes una columna 'content'

# Generar embeddings (para simplificar, usaremos TF-IDF o un modelo similar)
# Aquí utilizamos embeddings aleatorios; en producción, usa un modelo de embeddings adecuado
embeddings = np.random.rand(len(df), 768).astype('float32')  # Reemplaza con embeddings reales

# Crear índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

class Question(BaseModel):
    pregunta: str

@app.post("/responder/")
def responder_pregunta(q: Question):
    # Generar embedding para la pregunta
    # En producción, usa un modelo de embeddings
    pregunta_embedding = np.random.rand(1, 768).astype('float32')  # Reemplaza con embedding real
    
    # Buscar el documento más relevante
    D, I = index.search(pregunta_embedding, 1)
    doc_relevante = df.iloc[I[0][0]]['content']
    
    # Preparar inputs para el modelo de QA
    inputs = tokenizer.encode_plus(q.pregunta, doc_relevante, return_tensors='pt')
    input_ids = inputs["input_ids"].tolist()[0]
    
    # Obtener las respuestas
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    respuesta = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    return {"respuesta": respuesta, "documento_relevante": doc_relevante}

# Ejecutar el servidor con:
# uvicorn nombre_del_script:app --reload
```

#### **d. Mejoras sugeridas**

- **Generación de embeddings reales:** Utiliza modelos como `Sentence-BERT` para generar embeddings significativos de los documentos y las preguntas.
  
  ```python
  from sentence_transformers import SentenceTransformer
  model_embed = SentenceTransformer('all-MiniLM-L6-v2')
  embeddings = model_embed.encode(df['content'].tolist(), convert_to_tensor=False)
  embeddings = np.array(embeddings).astype('float32')
  ```
  
- **Gestión de grandes volúmenes de documentos:** Implementa índices más avanzados de FAISS para mejorar la velocidad de búsqueda.
- **Seguridad:** Implementa autenticación y autorización si la API será accesible públicamente.
- **Manejo de preguntas sin respuesta:** Añade lógica para manejar casos donde no se encuentra una respuesta adecuada.

---

### **Proyecto: Utilizar Transformers para detectar anomalías en datos de series temporales**

#### **a. Configuración inicial**

Emplearemos `transformers`, `torch`, y `pandas` para manejar y procesar datos de series temporales. Consideraremos usar modelos como `Time Series Transformer`.

#### **b. Instalación de dependencias**

```bash
pip install torch pandas matplotlib scikit-learn transformers
```

#### **c. Ejemplo de implementación**

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Definir el Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.float),
            torch.tensor(self.data[idx+self.seq_length], dtype=torch.float)
        )

# Cargar y preprocesar los datos
df = pd.read_csv('timeseries.csv')  # Supón una columna 'value'
values = df['value'].values.reshape(-1, 1)
scaler = StandardScaler()
values = scaler.fit_transform(values).flatten()

# Crear el dataset y dataloader
seq_length = 50
dataset = TimeSeriesDataset(values, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definir el modelo Transformer para series temporales
class TransformerAnomalyDetector(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerAnomalyDetector, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, src, tgt):
        src = self.embedding(src).permute(1, 0, 2)  # [seq, batch, dim]
        tgt = self.embedding(tgt).permute(1, 0, 2)
        output = self.transformer(src, tgt)
        output = self.fc_out(output).permute(1, 0, 2).squeeze(-1)
        return output

# Inicializar el modelo, pérdida y optimizador
model = TransformerAnomalyDetector()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Entrenamiento básico
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for src, tgt in dataloader:
        optimizer.zero_grad()
        output = model(src.unsqueeze(-1), tgt.unsqueeze(-1))
        loss = criterion(output, tgt.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

# Detección de anomalías
model.eval()
with torch.no_grad():
    predictions = []
    actual = []
    for src, tgt in dataloader:
        output = model(src.unsqueeze(-1), tgt.unsqueeze(-1))
        predictions.append(output.numpy())
        actual.append(tgt.numpy())
    predictions = np.concatenate(predictions)
    actual = np.concatenate(actual)
    mse = np.mean((predictions - actual) ** 2)
    print(f"Mean Squared Error: {mse}")

    # Definir un umbral para detectar anomalías
    threshold = mse * 1.5
    anomalies = np.abs(predictions - actual) > threshold

# Visualizar las anomalías
plt.figure(figsize=(15,5))
plt.plot(actual, label='Datos Reales')
plt.plot(predictions, label='Predicciones')
plt.scatter(np.where(anomalies)[0], actual[anomalies], color='red', label='Anomalías')
plt.legend()
plt.show()
```

#### **d. Consideraciones adicionales**

- **Embeddings contextuales:** Considera incorporar características adicionales en los embeddings, como estacionalidad o tendencias.
- **Evaluación de rendimiento:** Utiliza métricas como precisión, recall y F1-score para evaluar la detección de anomalías.
- **Modelos avanzados:** Explora arquitecturas más avanzadas como `Informer` o `Reformer` para manejar secuencias largas eficientemente.

---

### **Proyecto: Optimizar un modelo BERT para dispositivos móviles con técnicas de pruning, quantization y knowledge distillation**

#### **a. Configuración inicial**

Emplearemos `transformers`, `torch`, y `torchvision` para manejar el modelo BERT. Utilizaremos herramientas como `TorchScript` y `TensorFlow Lite` para la conversión.

#### **b. Instalación de dependencias**

```bash
pip install transformers torch torchvision
```

#### **c. Ejemplo de implementación**

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.utils import prune
from transformers import DistilBERTTokenizer, DistilBERTForSequenceClassification

# Cargar el modelo preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Aplicar pruning
parameters_to_prune = (
    (model.bert.encoder.layer[0].attention.self.query, 'weight'),
    (model.bert.encoder.layer[0].attention.self.key, 'weight'),
    (model.bert.encoder.layer[0].attention.self.value, 'weight'),
)

for module, param in parameters_to_prune:
    prune.l1_unstructured(module, param, amount=0.2)  # Podar el 20% de los pesos

# Eliminar las máscaras de pruning
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

# Aplicar quantization
model.eval()
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Knowledge Distillation (Usar DistilBERT como modelo estudiante)
student_model = DistilBERTForSequenceClassification.from_pretrained('distilbert-base-uncased')
student_tokenizer = DistilBERTTokenizer.from_pretrained('distilbert-base-uncased')

# Ejemplo simple de distillation (más complejo en producción)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# Preparar datos de ejemplo
texts = ["Este es un ejemplo.", "Otro texto para distilar."]
labels = [0, 1]
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=2)

# Definir optimizador y criterio
optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)
criterion = torch.nn.KLDivLoss(reduction='batchmean')

# Entrenar el modelo estudiante
student_model.train()
for epoch in range(3):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_outputs = model(input_ids, attention_mask=attention_mask).logits
        student_outputs = student_model(input_ids, attention_mask=attention_mask).logits
        loss = criterion(torch.log_softmax(student_outputs, dim=1), torch.softmax(teacher_outputs, dim=1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item()}")

# Guardar el modelo optimizado
student_model.save_pretrained('./distilbert_optimized')
student_tokenizer.save_pretrained('./distilbert_optimized')
```

#### **d. Conversión para dispositivos móviles**

**Usando TorchScript:**

```python
# Cargar el modelo optimizado
model = DistilBERTForSequenceClassification.from_pretrained('./distilbert_optimized')

# Convertir a TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('distilbert_optimized.pt')
```

**Usando TensorFlow Lite:**

```python
# Convertir el modelo a ONNX primero (requiere instalar torch.onnx)
import torch.onnx

dummy_input = torch.randint(0, 2000, (1, 128))
torch.onnx.export(model, dummy_input, "distilbert.onnx", opset_version=11)

# Luego convertir a TensorFlow Lite usando onnx-tf y tflite
# Requiere instalar onnx-tf y tflite_converter
```

#### **e. Consideraciones adicionales**

- **Evaluación del rendimiento:** Después de optimizar, evalúa el modelo en dispositivos móviles para medir el tiempo de inferencia y el uso de memoria.
- **Compatibilidad de frameworks:** Asegúrate de que el modelo convertido es compatible con frameworks como TensorFlow Lite o PyTorch Mobile según el dispositivo objetivo.
- **Mantenimiento de la precisión:** Ajusta las técnicas de optimización para minimizar la degradación de la precisión del modelo.

---

### **Proyecto: Crear una aplicación de resumen de documentos largos con un enfoque extractivo y abstractive en LLM**

#### **a. Configuración inicial**

Utilizaremos `transformers` para los modelos de resumen, `Flask` o `FastAPI` para la aplicación web, y `nltk` o `spacy` para el preprocesamiento.

#### **b. Instalación de dependencias**

```bash
pip install transformers flask nltk spacy
python -m spacy download en_core_web_sm
```

#### **c. Ejemplo de implementación**

**i. Resumen extractivo usando TextRank**

```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')

def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    word_tokens = [word_tokenize(sent.lower()) for sent in sentences]
    word_tokens = [[word for word in words if word.isalnum() and word not in stop_words] for words in word_tokens]
    
    # Crear matriz de similaridad
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = len(set(word_tokens[i]).intersection(set(word_tokens[j])))
    
    # Crear grafo y aplicar PageRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Ordenar las sentencias por puntuación
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = ' '.join([s for _, s in ranked_sentences[:num_sentences]])
    return summary
```

**ii. Resumen abstractive usando BART**

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# Cargar el modelo y el tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def abstractive_summary(text, max_length=130, min_length=30):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

**iii. Crear la aplicación web con Flask**

```python
from flask import Flask, request, render_template
import nltk

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        document = request.form['document']
        summary_type = request.form['type']
        if summary_type == 'extractive':
            summary = extractive_summary(document)
        else:
            summary = abstractive_summary(document)
        return render_template('index.html', summary=summary, document=document, type=summary_type)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

**iv. Crear la plantilla HTML (`templates/index.html`)**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Resumen de Documentos</title>
</head>
<body>
    <h1>Generador de Resúmenes</h1>
    <form method="post">
        <textarea name="document" rows="10" cols="80" placeholder="Pega tu documento aquí..."></textarea><br><br>
        <label for="type">Tipo de Resumen:</label>
        <select name="type" id="type">
            <option value="extractive">Extractivo</option>
            <option value="abstractive">Abstractive</option>
        </select><br><br>
        <button type="submit">Generar Resumen</button>
    </form>
    {% if summary %}
        <h2>Resumen {{ type.capitalize() }}:</h2>
        <p>{{ summary }}</p>
    {% endif %}
</body>
</html>
```

#### **d. Ejecución de la aplicación**

Guarda todos los scripts y la plantilla HTML en la estructura de carpetas adecuada y ejecuta:

```bash
python nombre_del_script.py
```

Visita `http://localhost:5000` en tu navegador para interactuar con la aplicación.

#### **e. Consideraciones adicionales**

- **Manejo de cocumentos largos:** Implementa segmentación de documentos y resúmenes por partes si el texto es demasiado largo para los modelos.
- **Interfaz de usuario mejorada:** Utiliza frameworks como `Bootstrap` para mejorar la apariencia de la interfaz.
- **Optimización del rendimiento:** Implementa caché para resúmenes frecuentes y optimiza el tiempo de generación utilizando procesamiento asíncrono.

---

### **Proyecto: Ajustar finamente un LLM con datos médicos para tareas como clasificación de documentos clínicos o extracción de información específica**

#### **a. Configuración inicial**

Utilizaremos modelos especializados como `BioBERT` y `transformers` para el ajuste fino. Además, emplearemos `pandas` y `scikit-learn` para manejar y evaluar los datos.

#### **b. Instalación de dependencias**

```bash
pip install transformers datasets scikit-learn pandas torch
```

#### **c. Ejemplo de implementación**

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# Cargar y preparar los datos
# Supón que tienes un CSV con columnas 'text' y 'label'
df = pd.read_csv('medical_documents.csv')
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Cargar el tokenizer y el modelo BioBERT
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=2)  # Ajusta 'num_labels' según tus clases

# Tokenizar los datos
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results_medical',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Definir las métricas de evaluación
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# Inicializar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
trainer.evaluate()

# Guardar el modelo ajustado
model.save_pretrained('./biobert_medical_finetuned')
tokenizer.save_pretrained('./biobert_medical_finetuned')
```

#### **d. Consideraciones adicionales**

- **Anonimización de datos:** Asegúrate de que los datos médicos estén anonimados para cumplir con regulaciones como HIPAA o GDPR.
- **Balance de clases:** Si ciertas categorías tienen menos muestras, utiliza técnicas de sobremuestreo o ajuste de pesos en la función de pérdida.
- **Validación cruzada:** Implementa validación cruzada para obtener una evaluación más robusta del modelo.
- **Interpretabilidad:** Utiliza técnicas como `LIME` o `SHAP` para interpretar las predicciones del modelo en contextos clínicos.

---

### **Proyecto: Crear un sistema de generación de texto condicional basado en estilos o temas específicos usando técnicas de control de generación en LLMs**

#### **a. Configuración inicial**

Utilizaremos modelos como `GPT-2` o `GPT-3` y técnicas de control mediante prompts y embeddings. Además, emplearemos `Flask` para la aplicación web.

#### **b. Instalación de dependencias**

```bash
pip install transformers flask
```

#### **c. Ejemplo de implementación**

**i. Implementación de control con prompts**

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Cargar el modelo y el tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, 
                             no_repeat_ngram_size=2, 
                             early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text
```

**ii. Crear la aplicación web con Flask**

```python
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        estilo = request.form['estilo']
        tema = request.form['tema']
        prompt = f"Estilo: {estilo}\nTema: {tema}\nTexto:"
        summary = generate_text(prompt)
        return render_template('index.html', summary=summary, estilo=estilo, tema=tema)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

**iii. Crear la plantilla HTML (`templates/index.html`)**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Generador de Texto Condicional</title>
</head>
<body>
    <h1>Generador de Texto Condicional</h1>
    <form method="post">
        <label for="estilo">Estilo:</label>
        <select name="estilo" id="estilo">
            <option value="Formal">Formal</option>
            <option value="Coloquial">Coloquial</option>
            <option value="Humorístico">Humorístico</option>
            <option value="Científico">Científico</option>
        </select><br><br>
        
        <label for="tema">Tema:</label>
        <input type="text" id="tema" name="tema" placeholder="Ej. Tecnología, Educación"><br><br>
        
        <button type="submit">Generar Texto</button>
    </form>
    {% if summary %}
        <h2>Texto Generado:</h2>
        <p>{{ summary }}</p>
    {% endif %}
</body>
</html>
```

#### **d. Mejoras sugeridas**

- **Uso de embeddings de control:** Implementa embeddings específicos para controlar el estilo y el tema de manera más precisa.
- **Interfaz de usuario mejorada:** Añade opciones para ajustar la longitud del texto, el tono y otros parámetros.
- **Filtrado de contenido:** Implementa filtros para evitar la generación de contenido inapropiado o sesgado.
- **Escalabilidad:** Optimiza el modelo para manejar múltiples solicitudes simultáneas utilizando servidores asíncronos o balanceadores de carga.

---

### **Proyecto: Optimizar un modelo Transformer con técnicas de pruning y quantization para su despliegue en dispositivos edge**

#### **a. Configuración inicial**

Emplearemos `transformers`, `torch`, y `torchvision` para manejar el modelo Transformer. Utilizaremos `torch.quantization` para la cuantización y `torch.nn.utils.prune` para la poda.

#### **b. Instalación de dependencias**

```bash
pip install transformers torch torchvision
```

#### **c. Ejemplo de implementación**

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.utils import prune
from torch.quantization import quantize_dynamic

# Cargar el modelo preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Aplicar pruning (poda)
parameters_to_prune = (
    (model.bert.encoder.layer[0].attention.self.query, 'weight'),
    (model.bert.encoder.layer[0].attention.self.key, 'weight'),
    (model.bert.encoder.layer[0].attention.self.value, 'weight'),
)

for module, param in parameters_to_prune:
    prune.l1_unstructured(module, param, amount=0.2)  # Podar el 20% de los pesos

# Eliminar las máscaras de pruning
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

# Aplicar quantization (cuantización dinámica)
model.eval()
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Guardar el modelo optimizado
quantized_model.save_pretrained('./bert_optimized')
tokenizer.save_pretrained('./bert_optimized')

# Opcional: Convertir a TorchScript para despliegue
scripted_model = torch.jit.script(quantized_model)
scripted_model.save('bert_optimized.pt')
```

#### **d. Despliegue en dispositivos edge**

**Usando PyTorch Mobile:**

```python
# Cargar el modelo TorchScript
scripted_model = torch.jit.load('bert_optimized.pt')

# Exportar el modelo para PyTorch Mobile
scripted_model.save('bert_optimized_mobile.pt')

# Integrar en una aplicación móvil usando PyTorch Mobile SDK
# Consulta la documentación oficial: https://pytorch.org/mobile/home/
```

#### **e. Consideraciones adicionales**

- **Evaluación del modelo:** Después de la optimización, evalúa el modelo en un conjunto de datos de prueba para asegurar que la precisión no se degrade significativamente.
- **Compatibilidad de hardware:** Asegúrate de que las optimizaciones son compatibles con el hardware específico del dispositivo edge.
- **Optimización adicional:** Considera técnicas como la cuantización por lotes (`batch quantization`) o el uso de aceleradores de hardware específicos.
- **Automatización del pipeline:** Implementa scripts que automaticen el proceso de poda, cuantización y conversión para facilitar futuras optimizaciones.

---

### **Consideraciones finales para todos los proyectos**

1. **Documentación y README.md:**
   - Asegúrate de mantener actualizado el archivo `README.md` en cada carpeta de entrega con detalles sobre los avances, instrucciones de ejecución y cualquier dependencia adicional.
   
2. **Control de versiones:**
   - Realiza commits frecuentes con mensajes claros que describan los cambios realizados.
   - Utiliza ramas (`branches`) para desarrollar nuevas funcionalidades sin afectar la rama principal (`main`).

3. **Entornos virtuales:**
   - Utiliza entornos virtuales (`venv`, `conda`) para aislar las dependencias de cada proyecto.
   
   ```bash
   python -m venv env
   source env/bin/activate  # En Linux/Mac
   env\Scripts\activate  # En Windows
   ```

4. **Gestión de dependencias:**
   - Crea un archivo `requirements.txt` para cada proyecto que liste todas las dependencias.
   
   ```bash
   pip freeze > requirements.txt
   ```

5. **Automatización de tareas:**
   - Considera usar scripts adicionales en Python o herramientas como `Makefile` para automatizar tareas recurrentes como el inicio de servidores, ejecución de tests, etc.

6. **Pruebas y validación:**
   - Implementa pruebas unitarias y de integración para asegurar la calidad del código.
   - Utiliza frameworks como `pytest` para facilitar la ejecución de pruebas.
   
   ```bash
   pip install pytest
   ```

7. **Seguridad:**
   - Implementa medidas de seguridad en las aplicaciones web, como validación de entradas, autenticación y autorización.
   - Utiliza HTTPS para comunicaciones seguras.

8. **Optimización y escalabilidad:**
   - Optimiza el rendimiento de los modelos y aplicaciones para asegurar una experiencia de usuario fluida.
   - Planifica la escalabilidad si esperas un crecimiento en el uso o en los datos manejados.

9. **Colaboración y buenas prácticas:**
   - Si trabajas en equipo, establece convenciones de codificación y utiliza herramientas como `pull requests` para revisar cambios.
   - Emplea linters (`flake8`, `black`) para mantener un código limpio y consistente.
   
   ```bash
   pip install flake8 black
   flake8 nombre_del_script.py
   black nombre_del_script.py
   ```

10. **Recursos y documentación:**
    - Familiarízate con la documentación de las librerías y herramientas que estás utilizando.
    - Consulta ejemplos y tutoriales para profundizar en funcionalidades específicas.

    - **Hugging Face Transformers:** [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
    - **FAISS:** [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
    - **FastAPI:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
    - **Flask:** [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
    - **PyTorch Mobile:** [https://pytorch.org/mobile/home/](https://pytorch.org/mobile/home/)
    - **Sentence-BERT:** [https://www.sbert.net/](https://www.sbert.net/)
    - **TorchScript:** [https://pytorch.org/docs/stable/jit.html](https://pytorch.org/docs/stable/jit.html)
    - **PEFT (Parameter-Efficient Fine-Tuning):** [https://github.com/huggingface/peft](https://github.com/huggingface/peft)


