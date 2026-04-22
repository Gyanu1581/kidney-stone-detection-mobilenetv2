import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
    return img
