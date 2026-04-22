import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Rescaling
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import MobileNetV2

def build_model():
    # 1. On-the-Fly Data Augmentation 
    # This automatically creates infinite, slightly rotated/zoomed variations 
    # of your original images during training to prevent memorization.
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ], name="data_augmentation")

    # 2. Load Pre-trained Base Model (Transfer Learning)
    # MobileNetV2 already knows how to detect edges, textures, and shapes.
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False, 
        weights='imagenet' 
    )
    
    # Freeze the pre-trained weights so we don't accidentally destroy them during the first few epochs of training.
    base_model.trainable = False 

    # 3. Build the final classification pipeline
    model = Sequential([
        # CRITICAL: MobileNetV2 strictly requires pixels scaled from -1 to +1
        Rescaling(1./127.5, offset=-1, input_shape=(224, 224, 3)),
        
        data_augmentation,
        base_model,
        
        # Shrink the feature maps down to a 1D vector
        GlobalAveragePooling2D(), 
        
        # Add a custom "brain" to learn what a kidney stone looks like
        Dropout(0.2), 
        Dense(128, activation='relu'),
        Dropout(0.5), # Heavy dropout to prevent overfitting on medical data
        
        # Final decision: 0 (Normal) or 1 (Stone)
        Dense(1, activation='sigmoid') 
    ])

    return model