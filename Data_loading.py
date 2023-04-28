import os
import cv2
import numpy as np
from pathlib import Path

def load_data(data_dir):
    images = []
    labels = []
    target_size = (256, 256)

    for category in ['healthy', 'parkinson']:
        label = 1 if category.lower() == "parkinson" else 0 
        folder_path = os.path.join(data_dir, category)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, target_size)  # Resize the image

            images.append(img_resized)
            labels.append(label)

    return np.array(images), np.array(labels)

def load_datasets(drawing_type):
    base_dir = Path("C:/Users/AnishMathew.Chacko/Desktop/Research Work")
    train_dir = base_dir / drawing_type / "training"
    test_dir = base_dir / drawing_type / "testing"

    x_train, y_train = load_data(train_dir)
    x_test, y_test = load_data(test_dir)

    return x_train, y_train, x_test, y_test

def load_augmented_datasets(drawing_type):
    base_dir1 = Path("C:/Users/AnishMathew.Chacko/Desktop/Research Work")
    train_dir1 = base_dir1/ f"{drawing_type}_augmented" / "training"
    test_dir1 = base_dir1 / f"{drawing_type}_augmented" / "testing"

    x_train, y_train = load_data(train_dir1)
    x_test, y_test = load_data(test_dir1)

    return x_train, y_train, x_test, y_test  
