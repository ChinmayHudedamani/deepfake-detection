import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MODEL_PATH = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\models\best_model.keras"
DATASET_PATH = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\data\processed"

IMG_SIZE = 224
BATCH_SIZE = 32

print("Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Evaluating...")
preds = model.predict(data)
y_pred = (preds > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(data.classes, y_pred, target_names=data.class_indices))

print("\nConfusion Matrix:")
print(confusion_matrix(data.classes, y_pred))
