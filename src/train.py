import os
# Mute annoying TensorFlow C++ backend warnings (Must be before importing tf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import callbacks
from model import build_model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# Smart Directory Finder
base_raw = "dataset_raw"
DATA_DIR = "dataset_raw/Original_Dataset"

# Automatically hunt for the correct Kaggle folder name
for root, dirs, files in os.walk(base_raw):
    if "Original_Dataset" in dirs:
        DATA_DIR = os.path.join(root, "Original_Dataset")
        break
    elif "Original Dataset" in dirs:
        DATA_DIR = os.path.join(root, "Original Dataset")
        break

if DATA_DIR is None:
    raise ValueError("Could not find the 'Original Dataset' folder! Run reset_data.py first.")

print(f"📁 Found clean data at: {DATA_DIR}")


# Dataset Loaders

# 1. Load Training Data
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 2. Load Validation Data (Pristine, completely unaugmented)
val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Performance optimization: Load data into RAM to keep the GPU/CPU fed faster
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


# Training Pipeline
def train():
    model = build_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # Callbacks to prevent overfitting and help the model un-stick itself
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    print("\n🚀 Starting training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr],
        verbose=2  # Keeps the terminal clean (one line per epoch)
    )

    # Ensure the save directory exists
    os.makedirs("model", exist_ok=True)
    model.save("model/kidney_stone_model.h5")
    print("\n✅ Training complete. Best model saved to 'model/kidney_stone_model.h5'.")

if __name__ == "__main__":
    train()