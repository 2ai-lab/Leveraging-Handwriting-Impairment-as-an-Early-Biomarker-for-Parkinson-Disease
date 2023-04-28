from Data_loading import load_datasets
from Data_loading import load_augmented_datasets
from Data_Aug import augment_data
from Data_Preprocessing import preprocess_data
import numpy as np
from keras.layers import Input, Conv2D, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1, l2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121
from keras.applications.xception import Xception


x_train_spiral, y_train_spiral, x_test_spiral, y_test_spiral = load_datasets('spiral')
x_train_wave, y_train_wave, x_test_wave, y_test_wave = load_datasets('wave')





spiral_output_dir = "spiral_augmented"
x_train_spiral_aug, y_train_spiral_aug = augment_data(x_train_spiral, y_train_spiral, num_augmentations=70, drawing_type='spiral', output_dir=spiral_output_dir)
x_test_spiral_aug, y_test_spiral_aug = augment_data(x_test_spiral, y_test_spiral, num_augmentations=20, drawing_type='spiral', is_test=True, output_dir=spiral_output_dir)

wave_output_dir = "wave_augmented"
x_train_wave_aug, y_train_wave_aug = augment_data(x_train_wave, y_train_wave, num_augmentations=70, drawing_type='wave', output_dir=wave_output_dir)
x_test_wave_aug, y_test_wave_aug = augment_data(x_test_wave, y_test_wave, num_augmentations=20, drawing_type='wave', is_test=True, output_dir=wave_output_dir)


x_train_spiral_aug, y_train_spiral_aug, x_test_spiral_aug, y_test_spiral_aug = load_augmented_datasets('spiral')

x_train_wave_aug, y_train_wave_aug, x_test_wave_aug, y_test_wave_aug = load_augmented_datasets('wave')

x_train_spiral_preprocessed, y_train_spiral_preprocessed = preprocess_data(x_train_spiral_aug, y_train_spiral_aug)
x_test_spiral_preprocessed, y_test_spiral_preprocessed = preprocess_data(x_test_spiral_aug, y_test_spiral_aug)

x_train_wave_preprocessed, y_train_wave_preprocessed = preprocess_data(x_train_wave_aug, y_train_wave_aug)
x_test_wave_preprocessed, y_test_wave_preprocessed = preprocess_data(x_test_wave_aug, y_test_wave_aug)

print(x_train_spiral_preprocessed.shape)
print(y_train_spiral_preprocessed.shape)
print(x_test_spiral_preprocessed.shape)
print(y_test_spiral_preprocessed.shape)

print(x_train_wave_preprocessed.shape)
print(y_train_wave_preprocessed.shape)
print(x_test_wave_preprocessed.shape)
print(y_test_wave_preprocessed.shape)






x_train_combined = np.concatenate((x_train_spiral_preprocessed, x_train_wave_preprocessed), axis=0)
y_train_combined = np.concatenate((y_train_spiral_preprocessed, y_train_wave_preprocessed), axis=0)


x_test_combined = np.concatenate((x_test_spiral_preprocessed, x_test_wave_preprocessed), axis=0)
y_test_combined = np.concatenate((y_test_spiral_preprocessed, y_test_wave_preprocessed), axis=0)


def convert_to_3_channels(images):
    return np.repeat(images[..., np.newaxis], 3, axis=-1)


x_train_3_channels = convert_to_3_channels(x_train_combined)
x_test_3_channels = convert_to_3_channels(x_test_combined)

# Creating  InceptionV3 model
# #creating  VGG model
 #creating resnet50 model
input_shape = x_train_3_channels.shape[1:]
# base_model = InceptionV3(include_top=False,
#                  weights="imagenet",
#  input_shape=input_shape)
base_model = VGG19(include_top=False,
                        weights="imagenet",
                        input_shape=input_shape)
# base_model = ResNet50(include_top=False,
#                          weights="imagenet",
#                          input_shape=input_shape)
# input_shape = x_train_3_channels.shape[1:]
# base_model = DenseNet121(include_top=False,
#                          weights="imagenet",
#                        input_shape=input_shape)
# base_model = Xception(include_top=False,
#                       weights="imagenet",
#                          input_shape=input_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
     layer.trainable = False


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(x_train_3_channels, y_train_combined, batch_size=32, epochs=10, validation_split=0.1)
# model.save_weights("Vgg16_model_weights.h5")



evaluation = model.evaluate(x_test_3_channels, y_test_combined)
print("Test loss:", evaluation[0])
print("Test accuracy:", evaluation[1])


plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# #  training and validation accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()







vgg19_model = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
x = vgg19_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=vgg19_model.input, outputs=predictions)


model.load_weights("C:\\Users\\AnishMathew.Chacko\\Desktop\\Research Work\\vgg19_model_weights.h5")


vgg19_model_train_preds = model.predict(x_train_3_channels)
vgg19_model_test_preds = model.predict(x_test_3_channels)

np.save("vgg19_model_train_preds.npy", vgg19_model_train_preds)
np.save("vgg19_model_test_preds.npy", vgg19_model_test_preds)