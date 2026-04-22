import os
# Mute TensorFlow C++ backend warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import numpy as np
import seaborn as sns

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# CRITICAL: Point exactly to the original images to avoid the ghost data
DATA_DIR = "dataset_raw/Original_Dataset" 
# DATA_DIR = "dataset"


print("Loading trained model...")
model = tf.keras.models.load_model("model/kidney_stone_model.h5")

print("Loading pristine validation dataset...")
# CRITICAL: 'seed=123' and 'shuffle=True' must match train.py exactly to ensure we evaluate on the exact same 20% validation split.
val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True 
)

print("Generating predictions batch by batch (this prevents shuffling mismatches)...")
y_true = []
y_pred_probs = []

# By predicting inside the loop, we guarantee the true label always matches the prediction
for images, labels in val_data:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0) 
    y_pred_probs.extend(preds.flatten())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

# Convert probabilities to a firm Yes/No decision (Threshold = 0.5)
y_pred_binary = (y_pred_probs > 0.5).astype(int)

# 1. Classification Report
print("\n" + "="*50)
print("FINAL CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred_binary, target_names=["Normal", "Stone"]))

# 2. ROC Curve & AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 5))

# Plot ROC
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='#FF5722', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='#3F51B5', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Falsely flagged as Stone)', fontsize=10)
plt.ylabel('True Positive Rate (Correctly flagged Stones)', fontsize=10)
plt.title('ROC Curve - Kidney Stone Detection', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# 3. Confusion Matrix
cm = confusion_matrix(y_true, y_pred_binary)

plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Normal", "Stone"], 
            yticklabels=["Normal", "Stone"],
            annot_kws={"size": 14})
plt.xlabel("Predicted Label", fontsize=10)
plt.ylabel("True Label", fontsize=10)
plt.title("Confusion Matrix", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()