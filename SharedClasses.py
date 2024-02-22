import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class CustomImageDataGenerator:
    def __init__(self, directory, image_width, image_height, batch_size, class_mode='categorical'):
        self.directory = directory
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.image_files = self.collect_image_files()

    def collect_image_files(self):
        image_files = {}

        for category in os.listdir(self.directory):
            category_dir = os.path.join(self.directory, category)
            if os.path.isdir(category_dir):
                image_files[category] = []
                for image_file in os.listdir(category_dir):
                    if image_file.endswith('.jpg'):
                        image_files[category].append(os.path.join(category_dir, image_file))

        return image_files

    def calculate_num_samples(self):
        return sum(len(files) for files in self.image_files.values())

    def generate_data(self, is_training=True):
        categories = os.listdir(self.directory)  # List of category folder names
        all_cases = []  # Collect all cases in all categories
        if is_training:  # Only augment training data
            datagen = ImageDataGenerator(
                rotation_range=45,
                width_shift_range=0.35,
                height_shift_range=0.35,
                zoom_range=0.35,
                horizontal_flip=True,
                fill_mode='nearest'
            )

        for category in categories:
            category_path = os.path.join(self.directory, category)
            image_files = os.listdir(category_path)
            guids = set()

            for image_file in image_files:
                if image_file.endswith('.jpg'):
                    guid, image_type = image_file.split('_')[:2]
                    guids.add(guid)

            all_cases.extend([(category, guid) for guid in guids])

        while True:
            random.shuffle(all_cases)  # Shuffle cases for better training performance
            batch_images = []
            batch_labels = []  # Initialize an empty list for batch labels

            for (category, guid) in all_cases:

                image_file = f"{guid}_close_up.jpg"
                image_path = os.path.join(self.directory, category, image_file)
                image = Image.open(image_path)
                image = image.resize((self.image_width, self.image_height))
                image_array = np.array(image) / 255.0
                if is_training:
                    image_array = datagen.random_transform(image_array)  # Apply transformations
                   
                batch_labels.append(category)
                batch_images.append(image_array)

                if len(batch_images) == self.batch_size:
                    batch_images = np.array(batch_images)
                    if self.class_mode == 'categorical':
                        # Convert labels to numerical format
                        label_to_index = {label: i for i, label in enumerate(categories)}
                        batch_labels = [label_to_index[label] for label in batch_labels]
                        batch_labels = tf.keras.utils.to_categorical(batch_labels, len(categories))

                    yield batch_images, batch_labels

                    # Clear the batch lists
                    batch_images = []
                    batch_labels = []  # Clear the batch labels
            
            # If there are not enough images left to form a full batch, discard them
            batch_images = []
            batch_labels = []

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, condition, verbose=0):
        super(CustomEarlyStopping, self).__init__()
        self.condition = condition
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if self.condition(logs):
            self.model.stop_training = True
            if self.verbose > 0:
                print(f"Custom early stopping triggered at epoch {epoch + 1}.")


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
