# 📊 Olortegui Lab 11 - Minería de Datos

Este repositorio contiene la implementación de utilidades para análisis de datos y modelado predictivo, incluyendo funciones para cargar, escalar, limpiar datos y entrenar modelos de regresión.

## 📁 Estructura del proyecto

```

olortegui\_lab11/
│
├── src/
│   └── utils.py              # Funciones principales: cargar datos, limpieza, modelado
│
├── tests/
│   └── test\_utils.py         # Pruebas unitarias con pytest
│
├── .venv/                    # Entorno virtual de Python (excluido en GitHub)
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo

````

## ⚙️ Requisitos

- Python 3.10 o superior (compatible con Python 3.13)
- pip

## 📦 Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/tu_usuario/olortegui_lab11.git
cd olortegui_lab11
````

2. Crea un entorno virtual y actívalo:

```bash
python -m venv .venv
# En Windows
.venv\Scripts\activate
# En macOS/Linux
source .venv/bin/activate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## 🧠 Funcionalidades principales

### 📥 `load_data_from_url(url: str) -> pd.DataFrame`

Carga un dataset desde una URL pública de Google Drive.

### 🧹 `remove_outliers(df, columns)`

Elimina outliers utilizando el rango intercuartílico (IQR).

### 📐 `scale_columns(df, columns)`

Escala columnas numéricas usando `StandardScaler`.

### 📈 `evaluate_model(X, y)`

Entrena un modelo de regresión lineal y devuelve el R² score mediante `cross_val_score`.

### 🤖 `train_final_model(X, y)`

Entrena un modelo de regresión lineal y devuelve el modelo y su rendimiento (`R²`).

---

## 🧪 Ejecución de tests

```bash
pytest tests/test_utils.py
```

Los tests cubren:

* Limpieza de datos
* Escalamiento
* Evaluación de modelo
* Entrenamiento final
* Manejo de errores y edge cases

---

## 🔗 Ejemplo de carga de datos

```python
from src.utils import load_data_from_url

URL = "https://drive.google.com/file/d/1JvFvM_AUeD-ikbVaZrYi4Q4RWe9sCFIv/view?usp=sharing"
df = load_data_from_url(URL)
print(df.head())
```

---

## 🧑‍💻 Autor

Kevin Olortegui
Laboratorio 11 de Minería de Datos - 2025
TECSUP
---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT.

```

---

¿Quieres que lo genere directamente como archivo o quieres agregar alguna sección más (por ejemplo, sobre contribuciones o referencias)?
```
