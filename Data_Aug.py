import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def augment_data(x, y, num_augmentations, drawing_type, is_test=False, output_dir=""):
    data_generator = ImageDataGenerator(rotation_range=360,
                                        width_shift_range=0.0,
                                        height_shift_range=0.0,
                                        horizontal_flip=True,
                                        vertical_flip=True)

    x_augmented = []
    y_augmented = []

    for i, label in enumerate(y):
        img = x[i]
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        aug_iter = data_generator.flow(img, batch_size=1, shuffle=True)
        
        for j in range(num_augmentations):
            aug_img = next(aug_iter)[0].astype('uint8')
            x_augmented.append(aug_img)
            y_augmented.append(label)

            # Save the augmented image
            if output_dir:
                label_dir = 'healthy' if label == 0 else 'parkinson'
                data_dir = 'training' if not is_test else 'testing'
                save_dir = os.path.join(output_dir, data_dir, label_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                filename = f"{save_dir}/aug_{i}_{j}.png"
                cv2.imwrite(filename, aug_img)

    return x_augmented, y_augmented


