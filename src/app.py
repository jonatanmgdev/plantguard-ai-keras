import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pandas as pd
import altair as alt

# Cargar modelo
model_path = os.path.join("..", "model", "best_plantGuardAI_model.keras")
model = tf.keras.models.load_model(model_path)

# Obtener nombres de clases desde el modelo
class_indices = {v: k for k, v in model.class_names.items()} if hasattr(model, 'class_names') else None

# Traducciones de clases al español
class_translations = {
    "Tomato___Bacterial_spot": "Mancha bacteriana",
    "Tomato___Early_blight": "Tizón temprano",
    "Tomato___Late_blight": "Tizón tardío",
    "Tomato___Leaf_Mold": "Moho de la hoja",
    "Tomato___Septoria_leaf_spot": "Mancha foliar por Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Ácaros araña (ácaro de dos manchas)",
    "Tomato___Target_Spot": "Mancha diana",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Virus del rizado amarillo de la hoja del tomate",
    "Tomato___Tomato_mosaic_virus": "Virus del mosaico del tomate",
    "Tomato___healthy": "Tomate saludable"
}

# Página de Streamlit
st.set_page_config(page_title="PlantGuardAI", layout="centered")
st.title("🌿 PlantGuardAI")
st.subheader("Clasificador de enfermedades de plantas")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen de la hoja", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen cargada", use_container_width=True)

    # Preprocesamiento
    img_resized = img.resize((256, 256))
    img_array = image.img_to_array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_batch)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Obtener nombres de clase
    if class_indices:
        class_names = [class_indices[i] for i in range(len(class_indices))]
    else:
        class_names = list(os.listdir(os.path.join("..", "data", "dataset", "train")))

    predicted_class_name = class_names[predicted_class]
    translated_name = class_translations.get(predicted_class_name, predicted_class_name)

    # Mostrar predicción
    st.markdown(f"### 🧠 Predicción: **{translated_name}** ({predicted_class_name})")

    # Crear DataFrame para gráfico
    class_probs = prediction[0]
    class_labels_es = [class_translations.get(name, name) for name in class_names]
    df = pd.DataFrame({
        'Clase': class_labels_es,
        'Probabilidad': class_probs
    })

    # Crear gráfico de barras personalizado con Altair
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Probabilidad:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Clase:N', sort='-x'),
        tooltip=['Clase', 'Probabilidad']
    ).properties(
        width=600,
        height=400,
        title="Distribución de probabilidades por clase"
    )

    st.altair_chart(chart, use_container_width=True)