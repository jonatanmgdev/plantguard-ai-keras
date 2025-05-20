import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Rutas
val_dir = os.path.join("..", "data", "dataset", "val")
model_path = os.path.join("..", "model", "best_plantGuardAI_model.keras")

# Cargar modelo
model = tf.keras.models.load_model(model_path)

# Obtener clases desde la estructura de carpetas
class_names = sorted(os.listdir(val_dir)) 
class_indices = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for name, idx in class_indices.items()}

# Métricas
y_true = []
y_pred = []

# Recorrer imágenes por clase
for class_name, true_label in class_indices.items():
    class_folder = os.path.join(val_dir, class_name)
    for fname in os.listdir(class_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_folder, fname)
            
            # Preprocesamiento
            img = image.load_img(img_path, target_size=(256, 256))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predicción
            predictions = model.predict(img_array)
            predicted_label = np.argmax(predictions, axis=1)[0]

            y_true.append(true_label)
            y_pred.append(predicted_label)

            print(f"{fname}: Predicción: {idx_to_class[predicted_label]}, Realidad: {class_name}")

# Resultados
print("\n--- Resultados Generales ---\n")
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2%}")

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()
