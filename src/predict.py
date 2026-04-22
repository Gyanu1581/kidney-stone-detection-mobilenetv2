import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model once when the script starts
model = tf.keras.models.load_model("model/kidney_stone_model.h5")

def get_last_conv_layer(model_to_search):
    """Dynamically finds the last Convolutional layer in the model."""
    for layer in reversed(model_to_search.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return model_to_search, layer.name
        # If using Transfer Learning, search inside the nested base model
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, tf.keras.layers.Conv2D):
                    return layer, inner_layer.name
    return None, None

def predict_image(image_path):
    # 1. Read and fix image colors (BGR to RGB)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Prepare for prediction
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Auto-detect if we need to scale manually (1/255)
    has_rescaling = any("rescaling" in layer.name.lower() for layer in model.layers)
    if not has_rescaling:
        img_array = img_array / 255.0

    # 2. Get Prediction & Format Label
    prediction = model.predict(img_array, verbose=0)[0][0]
    if prediction > 0.5:
        label = f"Stone Detected ({prediction*100:.1f}% Confidence)"
    else:
        label = f"Normal Kidney ({(1-prediction)*100:.1f}% Confidence)"

    # 3. Generate Grad-CAM Heatmap
    target_model, last_conv_layer_name = get_last_conv_layer(model)
    
    if target_model and last_conv_layer_name:
        try:
            # Map inputs to the last conv layer and the final output
            grad_model = tf.keras.models.Model(
                [target_model.inputs], 
                [target_model.get_layer(last_conv_layer_name).output, target_model.output]
            )
            
            with tf.GradientTape() as tape:
                inputs = img_array if target_model == model else img_array / 255.0 
                last_conv_layer_output, preds = grad_model(inputs)
                class_channel = preds[:, 0]

            # Calculate gradients 
            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Overlay gradients onto the feature map
            heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()

            # 4. Superimpose heatmap onto the original image
            heatmap = np.uint8(255 * heatmap)
            colormap = plt.get_cmap("jet") # Red = High impact, Blue = Low impact
            heatmap_colors = colormap(np.arange(256))[:, :3][heatmap]
            
            heatmap_colors = cv2.resize(heatmap_colors, (img.shape[1], img.shape[0]))
            superimposed_img = heatmap_colors * 0.4 + (img_rgb / 255.0)
            superimposed_img = np.clip(superimposed_img, 0, 1)
            
            output_path = "heatmap_result.jpg"
            plt.imsave(output_path, superimposed_img)
            return label, output_path
            
        except Exception as e:
            print(f"Heatmap generation bypassed: {e}")
            
    # Fallback if no heatmap could be generated
    return label, image_path