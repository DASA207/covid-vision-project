import os
import numpy as np
import tensorflow as tf
from keras import models, layers
from PIL import Image

# Define the Image_generator class
class Image_generator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, image_size, batch_size):
        self.file_paths = file_paths
        self.labels = labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.indices = np.arange(len(file_paths))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.file_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_indices_range = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_indices_paths = [self.file_paths[i] for i in batch_indices_range]
        batch_indices_labels = [self.labels[i] for i in batch_indices_range]
        return self.data_generator(batch_indices_paths, batch_indices_labels)

    def data_generator(self, batch_indices_paths, batch_indices_labels):
        images = []
        for i in batch_indices_paths:
            image = Image.open(i).convert('RGB')
            img = image.resize(self.image_size)
            image_array = np.array(img) / 255.0
            images.append(image_array)
        images = np.array(images)
        labels = np.array(batch_indices_labels)
        return images, labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Paths to the dataset
virus = r"C:\Users\harsh\Downloads\archive\COVID_IEEE\covid"
normal =r"C:\Users\harsh\Downloads\archive\COVID_IEEE\normal"

viral_path = [os.path.join(virus, i) for i in os.listdir(virus)]
normal_path = [os.path.join(normal, i) for i in os.listdir(normal)]

viral_label = np.array([0] * len(viral_path))
normal_label = np.array([1] * len(normal_path))

file_paths = np.concatenate((viral_path, normal_path), axis=0)
labels = np.concatenate((viral_label, normal_label), axis=0)

image_size = (128, 128)
batch_size = 32

training = Image_generator(file_paths, labels, image_size, batch_size)

# Define the CNN model
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(training, epochs=15)

# Save the model
cnn.save('covid_cnn_model.h5')

