import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# ==============================
# PATHS
# ==============================
DATASET_PATH = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\data\processed"
MODEL_SAVE_PATH = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\models\best_model.keras"

IMG_SIZE = 224
BATCH_SIZE = 32

# ==============================
# ADVANCED DATA AUGMENTATION
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ==============================
# HANDLE CLASS IMBALANCE
# ==============================
counts = np.bincount(train_data.classes)
total = np.sum(counts)

class_weights = {
    0: total / (2 * counts[0]),
    1: total / (2 * counts[1])
}

print("Class Weights:", class_weights)

# ==============================
# LOAD PRETRAINED MODEL
# ==============================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze MOST layers initially
for layer in base_model.layers:
    layer.trainable = False

# ==============================
# CUSTOM CLASSIFIER HEAD
# ==============================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# ==============================
# PHASE 1 TRAINING (Feature Learning)
# ==============================
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)

print("\n===== PHASE 1 TRAINING (Frozen Backbone) =====\n")

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights
)

# ==============================
# PHASE 2 FINE-TUNING (Deep Learning)
# ==============================
print("\n===== PHASE 2 FINE-TUNING =====\n")

# Unfreeze deeper layers
for layer in base_model.layers[-80:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # VERY LOW LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)

# ==============================
# CALLBACKS (IMPORTANT)
# ==============================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=3, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weights
)

# ==============================
# SAVE FINAL MODEL
# ==============================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)

print("\n✅ Advanced Model Training Complete!")
print("Best model saved at:", MODEL_SAVE_PATH)
