# Intrusion_Detection_System
Proyecto de Security Data Science enfocado en el analisis exploratorio del dataset **CIC-IDS2017** para deteccion de intrusiones.

## Descripcion
Este proyecto usa el dataset **CIC-IDS2017** del Canadian Institute for Cybersecurity (CIC), disponible oficialmente en:

- UNB CIC: https://www.unb.ca/cic/datasets/ids-2017.html
- Kaggle: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data

El notebook principal es [data_mining.ipynb](/c:/Users/javil/OneDrive/Documentos/U/Data%20Science/Intrusion_Detection_System/data_mining.ipynb) y contiene:

- descarga automatica del dataset si la carpeta `dataset/` no existe o no tiene archivos
- analisis exploratorio del conjunto de datos
- distribucion de clases
- revision de calidad de datos
- descripcion de caracteristicas clave para IDS
- entrenamiento y evaluacion de un `Decision Tree`
- entrenamiento y evaluacion de un `LSTM`
- experimento binario balanceado para comparar 2 modelos de machine learning y 2 de deep learning

La logica reusable del EDA esta en [cicids_eda.py](/c:/Users/javil/OneDrive/Documentos/U/Data%20Science/Intrusion_Detection_System/cicids_eda.py).
La comparacion balanceada `BENIGN` vs `ATTACK` esta en [cicids_balanced_binary.py](/c:/Users/javil/OneDrive/Documentos/U/Data%20Science/Intrusion_Detection_System/cicids_balanced_binary.py).

## Estructura del proyecto
- `data_mining.ipynb`: notebook principal del proyecto
- `cicids_eda.py`: funciones auxiliares para el analisis exploratorio
- `cicids_balanced_binary.py`: entrenamiento balanceado y evaluacion binaria de 2 modelos ML + 2 modelos DL
- `dataset/`: carpeta donde se guardan los CSV del dataset
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto

## Requisitos
- Python 3.13 recomendado
- `pip`
- Acceso a internet si quieres descargar el dataset desde Kaggle con la primera celda
- Credenciales de Kaggle si vas a usar la descarga automatica

## Librerias necesarias
Las dependencias estan en [requirements.txt](/c:/Users/javil/OneDrive/Documentos/U/Data%20Science/Intrusion_Detection_System/requirements.txt):

- `kagglehub`: descarga el dataset desde Kaggle
- `numpy` y `pandas`: manipulacion y analisis de datos tabulares
- `matplotlib` y `seaborn`: visualizaciones del EDA
- `scikit-learn`: ranking de caracteristicas e importancia de variables
- `torch`: entrenamiento del modelo LSTM
- `ipykernel`, `notebook` y `jupyterlab`: ejecucion del notebook en Jupyter

## Opcion 1: Descargar el dataset automaticamente
El notebook ya incluye una celda que:

1. crea la carpeta `dataset/` si no existe
2. descarga el dataset desde Kaggle si no hay archivos
3. evita volver a descargar si los CSV ya estan presentes

Para usar esa opcion necesitas tener configuradas tus credenciales de Kaggle.

### Credenciales de Kaggle
Puedes configurarlas con tu archivo `kaggle.json`.

En Windows suele colocarse en:
- `%USERPROFILE%\\.kaggle\\kaggle.json`

En macOS suele colocarse en:
- `~/.kaggle/kaggle.json`

En macOS tambien es recomendable ejecutar:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Opcion 2: Descargar el dataset manualmente
Si no quieres configurar Kaggle, puedes descargar el dataset manualmente desde:

- https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data

Despues:

1. crea una carpeta llamada `dataset` en la raiz del proyecto si no existe
2. extrae ahi los archivos CSV del dataset
3. ejecuta el notebook normalmente

Si la carpeta `dataset/` ya tiene los archivos, la celda de descarga no hara nada.

## Instalacion en Windows
Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name intrusion-detection-system --display-name "Intrusion Detection System"
```

Si PowerShell bloquea la activacion del entorno virtual, ejecuta primero:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Instalacion en macOS
Abre Terminal en la carpeta del proyecto y ejecuta:

```bash
python3 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name intrusion-detection-system --display-name "Intrusion Detection System"
```

## Como correr el notebook
Con el entorno virtual activado, puedes abrir el notebook con cualquiera de estas opciones:

### Opcion A: Jupyter Notebook
Windows:

```powershell
jupyter notebook data_mining.ipynb
```

macOS:

```bash
jupyter notebook data_mining.ipynb
```

### Opcion B: JupyterLab
Windows:

```powershell
jupyter lab
```

macOS:

```bash
jupyter lab
```

Luego abre [data_mining.ipynb](/c:/Users/javil/OneDrive/Documentos/U/Data%20Science/Intrusion_Detection_System/data_mining.ipynb) y selecciona el kernel `Intrusion Detection System` si Jupyter te lo pide.

## Orden recomendado de ejecucion
Dentro del notebook:

1. ejecuta la celda de descarga del dataset
2. ejecuta la celda `artifacts = run_eda(DATASET_DIR)`
3. revisa la tabla `feature_report`
4. ejecuta el experimento binario balanceado `BENIGN` vs `ATTACK`
5. revisa la tabla final de comparacion de modelos

## Estrategia contra el desbalance
El dataset CIC-IDS2017 esta desbalanceado: la clase `BENIGN` domina el volumen y varias familias de ataque tienen muy pocas filas. Para evitar una evaluacion enganosa, el proyecto usa esta regla:

- `train`, `validation` y `test` se separan primero con `stratify`.
- Solo el entrenamiento se balancea.
- `validation` y `test` conservan la distribucion original del trafico.
- Los modelos de machine learning usan `class_weight`.
- Los modelos deep learning usan `WeightedRandomSampler` para formar batches balanceados.
- La seleccion se hace con `F1 macro`, `balanced_accuracy` y metricas especificas de la clase `ATTACK`, no solo con `accuracy`.

El experimento principal para detectar intrusos usa `Label_Binary`:

- `0 = BENIGN`
- `1 = ATTACK`

Los cuatro modelos comparados son:

- `decision_tree_balanced`
- `random_forest_balanced`
- `mlp_weighted_sampler`
- `lstm_weighted_sampler`

## Salidas esperadas
Al ejecutar el notebook deberias obtener:

- tablas de resumen del dataset
- distribucion de clases por archivo
- revision de nulos, infinitos y duplicados
- ranking de caracteristicas clave
- descripciones de variables importantes para el analisis IDS
- comparacion binaria balanceada de 2 modelos ML y 2 modelos DL
- resumen interpretativo final para apoyar el informe

## Notas
- El primer analisis puede tardar varios minutos porque el dataset es grande.
- El notebook usa `cicids_eda.py`, asi que ambos archivos deben permanecer en la raiz del proyecto.
- Si ya descargaste el dataset una vez, no necesitas volver a descargarlo en ejecuciones posteriores.
