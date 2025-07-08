### **Examen Final CC0C2**

#### **1. Requisitos generales comunes**

Todos los proyectos comparten una serie de **condiciones y pautas** obligatorias:

1. **Individuales**: Cada estudiante entrega su propio repositorio, sin colaboración de pares.
2. **Código principal**:

   * Ubicado en `src/`.
   * Puede escribirse en Python puro o apoyarse en frameworks ML como PyTorch, TensorFlow o JAX.
   * Se excluyen celdas de prueba o benchmarking; éstas van en carpetas separadas.
3. **Pruebas**:

   * Carpeta `tests/`.
   * Al menos **5 pruebas unitarias** y/o funcionales que cubran los componentes críticos.
   * Uso de `pytest` y `pytest-cov` para medición de cobertura mínima aceptable (> 70 %).
4. **Benchmarks**:

   * Carpeta `benchmarks/`.
   * Scripts o notebooks reproducibles que midan tiempo de ejecución, uso de memoria y/o métricas de calidad (BLEU, F1, perplexity, etc.).
   * Debe incluir instrucciones para repetir las mediciones (ej. `bash run_bench.sh` o `python bench.py`).
   * 
5. **Cuaderno de exposición**:

   * Archivo único `exposicion.ipynb` en la raíz.
   * Entre 15 y 25 celdas: título, contexto, fragmentos de código demostrativo, gráficos y conclusiones parciales.
   * **Comentarios y cadenas de texto** en español, sin símbolos extraños (como comillas tipográficas de ChatGPT) o íconos, etc.
     
6. **Video de presentación**:

   * Duración mínima 5 minutos, con audio en español.
   * Debe explicar:

     * Motivación y problema planteado.
     * Diseño de la solución (arquitectura, algoritmos, librerías).
     * Demostración rápida de resultados.
     * Reflexión sobre limitaciones y posibles extensiones.
   * Enlace al video en el `README.md`.
7. **Preguntas orales**:

   * 50 % de la nota se basa en la defensa en clase.
   * Se valorará la claridad al responder sobre teoría, implementación y resultados.
8. **Penalizaciones**:

   * Código en otro idioma, video sin audio, pruebas ausentes, benchmarks irreproducibles o cuaderno adicional implican **cero** en la sección correspondiente.


#### **2. Estructura del repositorio**

```plaintext
proyectoX/                    <- Raíz del repositorio
├── src/                      <- Implementación principal (≥ 800 LDC)
│   ├── __init__.py
│   ├── modelo.py             <- Definición de la arquitectura
│   ├── datos.py              <- Carga y preprocesamiento
│   ├── decodificadores.py    <- Módulos reusables (solo en proyecto 1)
│   ├── entrenamiento.py      <- Rutinas de entrenamiento / fine-tuning
│   ├── inferencia.py         <- Funciones de generación / evaluación
│   └── utils.py              <- Funciones auxiliares (logging, métricas)
│
├── tests/                    <- Pruebas (pytest)
│   ├── test_modelo.py
│   ├── test_datos.py
│   ├── test_decodificadores.py
│   └── test_utils.py
│
├── benchmarks/               <- Scripts o notebooks de benchmarking
│   ├── run_bench.sh          <- Script bash "todo en uno"
│   ├── benchmark.py          <- Código para medir tiempo/uso memoria
│   └── bench_results.csv     <- Ejemplo de salida
│
├── exposicion.ipynb          <- Único cuaderno evaluable
├── requirements.txt          <- Dependencias exactas (pip freeze)
├── README.md                 <- Instrucciones de ejecución y links al video
└── .gitignore
```

**2.1. `src/`**

* **Modularidad**: Cada clase o función relevante debe vivir en su propio archivo, evitando módulos monolíticos.
* **Tipado y docstrings**:

  ```python
  def traducir(texto: str, max_len: int = 50) -> List[str]:
      """
      Genera la traducción del texto de entrada.

      Args:
          texto (str): Oración en inglés.
          max_len (int): Longitud máxima de la traducción.

      Returns:
          List[str]: Tokens generados.
      """
      ...
  ```
* **Logging**: Usa la librería estándar `logging` para informar de etapas clave (carga de datos, inicio de entrenamiento, checkpoints guardados).
* **Configuración**: Parámetros (rutas de datos, batch size, learning rate) deben definirse en un archivo de configuración (`config.py`) o aceptarse por línea de comandos usando `argparse` o `hydra`.

**2.2. `tests/`**

* **Cobertura mínima**: 80 % sobre el código en `src/`.
* **Tipos de prueba**:

  * *Unitarias* para funciones puras (p. ej. cálculo de máscara causal).
  * *Integración* para flujos completos (p. ej. paso de datos → modelo → métrica).
* **Fixtures**: Uso de `@pytest.fixture` para crear datos dummy o modelos ligeros en memoria.

**2.3. `benchmarks/`**

* **Reproducibilidad**:

  * Fijar semillas (`random`, `numpy`, `torch.manual_seed`).
  * Documentar versiones de hardware (tipo de GPU) y software (versión de Python, librerías).
* **Métricas**:

  * **Tiempo**: P50, P95, P99 de generación o de un batch de validación.
  * **Memoria**: Uso de VRAM o RAM en pico.
  * **Calidad**: BLEU, ROUGE, F1, perplexity u otras según el proyecto.
* **Formato**: CSV o JSON con columnas claras: `experimento`, `parametro`, `valor`, `tiempo_ms`, `memoria_mb`.

**2.4. `exposicion.ipynb`**

* **Página inicial**: Título, autor, fecha y objetivos del proyecto.
* **Secciones mínimas**:

  1. **Introducción**: Motivación y contexto teórico.
  2. **Implementación**: Fragmentos de código clave (no todo).
  3. **Resultados**: Gráficos generados con Matplotlib (sin especificar colores).
  4. **Análisis**: Breve interpretación de las curvas o tablas.
* **Buenas prácticas**:

  * Evitar celdas de instalación (`%pip install`).
  * No incluir datos pesados (usar enlaces o subsampling).

**2.5. `README.md`**

Debe contener:

* **Descripción breve** de cada carpeta y del flujo de ejecución:

  ```markdown
  ## Cómo ejecutar
  1. Clonar el repositorio.
  2. `pip install -r requirements.txt`
  3. Ejecutar pruebas: `pytest -q --cov=src`
  4. Correr benchmarks: `bash benchmarks/run_bench.sh`
  5. Abrir `exposicion.ipynb` para la demo.
  6. Enlace al video: https://...
  ```
* **Requisitos de hardware**: GPU mínima o CPU aceptable.
* **Licencia** y **autor**.


#### **3. Detalles de cada proyecto**

A continuación se amplía el **Proyecto 1** como ejemplo detallado de cómo deben planificarse las ocho propuestas. Para los demás, se seguirá un patrón similar, adaptando los módulos y métricas.

**3.1. Proyecto 1: Decodificadores avanzados para Seq2Seq**

**3.1.1. Objetivos específicos**

1. Implementar cinco estrategias de decodificación:

   * Greedy
   * Beam Search
   * Top-k sampling
   * Top-p (nucleus sampling)
   * Diverse Beam Search
2. Contrastar calidad de traducción (BLEU) frente a latencia y uso de memoria.
3. Mantener un diseño modular que facilite añadir nuevas variantes.

**3.1.2. Estructura detallada**

```plaintext
proyecto1_seq2seq/
├── src/
│   ├── __init__.py
│   ├── modelo_seq2seq.py        <- Encoder-Decoder Transformer
│   ├── decoders/
│   │   ├── greedy.py            <- Clase GreedyDecoder
│   │   ├── beam.py              <- Clase BeamSearchDecoder
│   │   ├── topk.py              <- Clase TopKSampler
│   │   ├── topp.py              <- Clase TopPSampler
│   │   └── diverse_beam.py      <- Clase DiverseBeamDecoder
│   ├── datos.py                 <- Carga de Tatoeba/UH
│   └── evaluation.py            <- Funciones BLEU, tiempo, memoria
│
├── tests/
│   ├── test_greedy.py
│   ├── test_beam.py
│   └── test_evaluation.py
│
├── benchmarks/
│   ├── run_bench.sh
│   ├── bench_translation.py     <- Traducción de 1 000 pares y medición
│   └── results/
│       ├── bleu_vs_latency.csv
│       └── memoria_vs_len.csv
│
├── exposicion.ipynb
├── requirements.txt
└── README.md
```

**3.1.3. Contenido de `src/`**

* **`modelo_seq2seq.py`**

  ```python
  class TransformerSeq2Seq(nn.Module):
      def __init__(self, config):
          super().__init__()
          # Definir encoder, decoder y máscaras
  ```
* **`decoders/beam.py`**

  ```python
  class BeamSearchDecoder:
      def __init__(self, model, beam_width: int = 5):
          self.model = model
          self.beam_width = beam_width
      def decode(self, src: Tensor) -> List[str]:
          # Algoritmo de beam search clásico
  ```
* **`evaluation.py`**

  * Función `calcular_bleu(hypotheses, referencias)`.
  * Decorador `@measure_time` para cronometrar funciones.
  * Función `memory_usage()` usando `psutil`.

**3.1.4. Pruebas (`tests/`)**

* Verificar que cada decoder retorna siempre la misma secuencia al fijar `torch.manual_seed(0)`.
* Probar que `BeamSearchDecoder(beam_width=1)` equivale a `GreedyDecoder`.

**3.1.5. Benchmarks (`benchmarks/`)**

* Script `run_bench.sh` que:

  1. Activa entorno VirtualEnv.
  2. Ejecuta `bench_translation.py`.
  3. Guarda resultados en `benchmarks/results/`.
* Notebook `notebook_bench.ipynb` con visualizaciones:

  * BLEU vs latencia (línea).
  * Uso de memoria vs longitud de la oración (scatter).

**3.1.6. Cuaderno de exposición**

* Gráfico comparativo con cinco curvas (calidad vs latencia).
* Tabla resumen con valores medios y desvío estándar.
* Notas en español explicando cada pico y posible causa.


#### **4. Extensión a los demás proyectos**

Para los proyectos 2 a 8, **repite el mismo nivel de detalle**:

1. **Estructura de `src/`**:

   * Módulos bien separados (e.g., `lora.py`, `adapters.py`, `pretrain.py`, `quant.py`, `rag.py`).
   * Uso de clases y funciones con responsabilidades únicas.
2. **`tests/`**:

   * Pruebas específicas al dominio (e.g., convergencia para LoRA, máscara causal para GPT, integridad de índices FAISS).
   * Fixtures para datos sintéticos y reales.
3. **`benchmarks/`**:

   * Scripts que midan métricas propias (F1, perplexity, ROUGE, latencia p50/p95).
   * Plantillas de CSV/JSON uniformes para facilitar comparaciones.
4. **`exposicion.ipynb`**:

   * Una sola narrativa coherente: planteamiento, metodología, resultados y discusión breve.
   * Gráficos generados en el cuaderno con Matplotlib (sin colorear manualmente).
5. **`README.md`**:

   * Explicación paso a paso de instalación y ejecución.
   * Enlace al video y breve resumen de los hallazgos.

#### **5. Buenas prácticas avanzadas**

* **Control de versiones**:

  * Cada entrega debe hacerse en un **tag** de Git (por ejemplo, `v1.0-entrega`).
  * Commits atómicos y descriptivos (`"Añade clase TopPSampler con test de determinismo"`).
* **CI/CD (opcional)**:

  * Un workflow de GitHub Actions que ejecute `pytest` y los benchmarks básicos en `push`.
* **Gestión de dependencias**:

  * `requirements.txt` con versiones fijas.
  * Instrucciones para `conda` si se prefiere.
* **Documentación**:

  * Docstrings en todos los métodos públicos.
  * Comentarios concisos en español, explicando "por qué" además de "qué".


#### **6. Consejos para maximizar la nota**

1. **Pruebas exhaustivas**: Cubre casos límite (oraciones vacías, secuencias muy largas, tamaños de batch variables).
2. **Benchmarks claros**: Presenta tablas resumidas en el cuaderno y exporta resultados completos en CSV.
3. **Exposición pulida**: El video debe fluir como una charla técnica, con transiciones suaves entre teoría, código y demos.
4. **Defensa preparada**: Conoce a fondo cada decisión de diseño y los trade-offs (p. ej. ¿por qué elegir ALiBi sobre sinusoidal?).

> **Resultado esperado:** Un repositorio organizado, reproducible y documentado, con implementaciones limpias y modulares, pruebas sólidas y análisis de rendimiento transparente. Con ello, superarás las expectativas del curso y asegurarás una evaluación óptima.

#### **Evaluación**

| Criterio                               | Puntos | Descripción                                                                                                     |
| -------------------------------------- | :----: | --------------------------------------------------------------------------------------------------------------- |
| Trabajo en el repositorio              |    5   | Incluye `src, `tests/`, `benchmarks/` y **un** `exposicion.ipynb` en español, sin símbolos extra. |
| Video de presentación (> 5 min, audio) |    5   | Explicación clara de motivación, diseño, demo de resultados y reflexión; **audio obligatorio**.                 |
| Ronda de preguntas en clase            |   10   | Defensa individual: dominio teórico, implementación, análisis de resultados y preguntas durante la sesión oral.            |

