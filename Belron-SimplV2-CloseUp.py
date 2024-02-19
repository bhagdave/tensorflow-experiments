import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import random
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import DepthwiseConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
import datetime
from tensorflow.keras.callbacks import TensorBoard
os.environ['TF_GRPC_TIMEOUT'] = '3600'  # Set it to 1 hour (3600 seconds)

image_folder = './images-new'
image_height = 300
image_width = 300
model_name = 'belron-simple-closeup'
batch_size = 8
num_classes = 4
num_epochs = 500
conv_1_units = 32
conv_2_units = 64
conv_3_units = 128
dropout_rate = 0.5
dense_1_units = 256 
early_stopping = 33
steps_per_epoch = 100
learning_rate = 0.001
validation_steps = 100
l2_regularization = 0.001

log_dir = "logs/fit/closeup-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


def scheduler(epoch, lr):
    if epoch < 16:
        return learning_rate
    elif epoch < 76:
        return learning_rate * 0.1
    elif epoch < 200:
        return learning_rate * 0.01
    else:
        return learning_rate * 0.001

class CustomImageDataGenerator:
    def __init__(self, directory, image_width, image_height, batch_size=batch_size, class_mode='categorical'):
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
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
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

# Define your custom condition function
def custom_condition(logs):
    # You can define your condition based on loss, accuracy, or any other metric
    return logs.get('val_loss') < 0.2  # Example: Stop if loss is less than 0.2

# Create the custom callback
custom_early_stopping = CustomEarlyStopping(condition=custom_condition, verbose=1)

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# Initialize the CustomImageDataGenerator for training and validation
train_generator = CustomImageDataGenerator(os.path.join(image_folder, 'train/'), image_width, image_height, batch_size=batch_size)
validation_generator = CustomImageDataGenerator(os.path.join(image_folder, 'validate/'), image_width, image_height, batch_size=batch_size)

def model_builder():
    print("Building model")
    model = Sequential()

    model.add(SeparableConv2D(conv_1_units, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(conv_2_units, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(conv_3_units, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    # Replacing Flatten with GlobalAveragePooling2D
    model.add(GlobalAveragePooling2D())

    model.add(Dense(dense_1_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = model_builder()
model.summary()
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy', f1_score]
)

checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_accuracy', save_best_only=True, mode='auto')
# Define the early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping, verbose=1, mode='auto', restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# model.fit_generator(train_generator, epochs=num_epochs, validation_data=validation_generator, callbacks=[early_stopping, learning_rate_callback])

print("Training model")
# Fit the model
model.fit(
    train_generator.generate_data(),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator.generate_data(),
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[early_stopping, learning_rate_callback, checkpoint, custom_early_stopping, tensorboard_callback],
)

# Save the model
model.save(f"{model_name}.keras")

