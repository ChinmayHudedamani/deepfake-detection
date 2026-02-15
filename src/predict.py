import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ===== PATHS =====
MODEL_PATH =  r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\models\best_model.keras"

IMG_PATH = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\data\real\real_00001.jpg"

IMG_SIZE = 224

# ===== Load Model =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Load Image =====
img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ===== Predict =====
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("✅ REAL FACE")
else:
    print("🚨 DEEPFAKE DETECTED")

print(f"Confidence: {prediction:.4f}")
