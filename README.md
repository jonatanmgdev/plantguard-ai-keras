# PlantGuardAI 🌿 - Clasificación de Enfermedades en Hojas de Tomate - KERAS VERSION

Este proyecto implementa un modelo de Deep Learning para la **detección multiclase** a partir de imágenes de hojas, utilizando una arquitectura preentrenada como `DenseNet121`.

## 📁 Estructura del Proyecto

```plaintext
plantGuard-AI/
├── data/
│   ├── train/
│   │   ├── Tomato___Bacterial_spot/
│   │   └── Tomato___Early_blight/
│   │   └── Tomato___healthy/
│   │   └── Tomato___Late_blight/
│   │   └── Tomato___Leaf_Mold/
│   │   └── Tomato___Septoria_leaf_spot/
│   │   └── Tomato___Spider_mites Two-spotted_spider_mite/
│   │   └── Tomato___Target_Spot/
│   │   └── Tomato___Tomato_mosaic_virus/
│   │   └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   └── val/
│   │   ├── Tomato___Bacterial_spot/
│   │   └── Tomato___Early_blight/
│   │   └── Tomato___healthy/
│   │   └── Tomato___Late_blight/
│   │   └── Tomato___Leaf_Mold/
│   │   └── Tomato___Septoria_leaf_spot/
│   │   └── Tomato___Spider_mites Two-spotted_spider_mite/
│   │   └── Tomato___Target_Spot/
│   │   └── Tomato___Tomato_mosaic_virus/
│   │   └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│
├── model/
│   └── best_plantGuardAI_model.keras
│
├── train.py          # Entrenamiento del modelo
├── predict.py        # Predicciones sobre el set de prueba
└── README.md         # Este archivo
```

## 🚀 Requisitos

Instala los requerimientos necesarios ejecutando:

```bash
python -m venv venv

pip install -r requirements.txt
```

El dataset contiene 10 clases diferentes de hojas de tomate.
Link de descarga del dataset para el entrenamiento y test:
https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf/data


## 🧠 Entrenamiento
Para entrenar el modelo desde cero, ejecuta:
```bash
python train.py
```
Esto hará lo siguiente:
- Cargará las imágenes desde data/train.
- Aplicará data augmentation.
- Usará DenseNet121
- Entrenará el modelo con early stopping y reducción de tasa de aprendizaje.
- Guardará el mejor modelo en model/best_plantGuardAI_model.keras.
- Realizará fine-tuning de las últimas capas del modelo base.


##  📊 Evaluación
Durante el entrenamiento (train.py), se imprime:
- Matriz de Confusión
- Reporte de Clasificación (Precision, Recall, F1-Score) para:
```plaintext
Tomato___Bacterial_spot - Tomate – Mancha bacteriana
Tomato___Early_blight -	Tomate – Tizón temprano
Tomato___Late_blight - Tomate – Tizón tardío
Tomato___Leaf_Mold - Tomate – Moho foliar
Tomato___Septoria_leaf_spot	Tomate – Mancha foliar por Septoria
Tomato___Spider_mites Two-spotted_spider_mite - Tomate – Ácaros (araña roja de dos puntos)
Tomato___Target_Spot - Tomate – Mancha diana
Tomato___Tomato_Yellow_Leaf_Curl_Virus	Tomate – Virus del rizado amarillo de la hoja de tomate
Tomato___Tomato_mosaic_virus - Tomate – Virus del mosaico del tomate
Tomato___healthy - Tomate – Planta sana
```

##  🔍 Predicción
```bash
python predict.py
```
Esto realizará inferencias sobre las imágenes en data/test, mostrándote por consola la clase predicha y la real, y calculará la precisión global (accuracy).

## 📈 Resultados

El modelo alcanza una **precisión del 94%** en el conjunto de validación, con un F1-Score promedio de **0.94**. A continuación, un ejemplo de las métricas por clase:

```plaintext
- Tomato___Bacterial_spot: F1 = 0.97
- Tomato___Late_blight: F1 = 0.95
- Tomato___Tomato_mosaic_virus: F1 = 0.98
- Tomato___Spider_mites: F1 = 0.92
- Tomato___healthy: F1 = 0.96
```

La matriz de confusión y el reporte completo se generan automáticamente tras el entrenamiento.


## 📈 Interfaz Web con Streamlit
Este proyecto incluye una aplicación web interactiva para la clasificación de enfermedades en hojas de tomate usando Streamlit.
- Carga de imagen: Permite subir imágenes en formato JPG, JPEG o PNG de hojas de tomate.
- Procesa la imagen y utiliza el modelo best_plantGuardAI_model.keras para predecir la enfermedad presente.

## 🧑‍💻 Autor
Jonatan Montesdeoca González

## 📄 Licencia
Este proyecto está bajo la Licencia MIT.