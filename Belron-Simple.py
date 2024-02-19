import os
import tensorflow as tf
import random
import json
from SharedClasses import f1_score, CustomEarlyStopping, CustomImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import DepthwiseConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image

image_folder = './images-new/close_up'
image_height = 256
image_width = 256
model_name = 'belron-simplei-256'
batch_size = 8
num_classes = 2
num_epochs = 100 
conv_1_units = 160
dropout_rate = 0.2
dense_1_units = 160 
dense_2_units = 192 
dense_3_units = 192
dense_4_units = 96
early_stopping = 10
steps_per_epoch = 462
learning_rate = 0.0001
validation_steps = 20

def scheduler(epoch, lr):
    if epoch < 20:
        return learning_rate
    elif epoch < 35:
        return learning_rate * .5
    elif epoch < 50:
        return learning_rate * .1
    else:
        return learning_rate * .05


# Define your custom condition function
def custom_condition(logs):
    # You can define your condition based on loss, accuracy, or any other metric
    return logs.get('val_loss') < 0.2  # Example: Stop if loss is less than 0.2

# Create the custom callback
custom_early_stopping = CustomEarlyStopping(condition=custom_condition, verbose=1)

# Initialize the CustomImageDataGenerator for training and validation
train_generator = CustomImageDataGenerator(os.path.join(image_folder, 'train/'), image_width, image_height, batch_size=batch_size)
validation_generator = CustomImageDataGenerator(os.path.join(image_folder, 'validate/'), image_width, image_height, batch_size=batch_size)

def model_builder():
    print("Building model")
        # ... other layers ...
    model = Sequential()
    model.add(Conv2D(conv_1_units, (3,3), activation='relu', input_shape=(image_height, image_width, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(DepthwiseConv2D((3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(DepthwiseConv2D((3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(DepthwiseConv2D((3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(DepthwiseConv2D( (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(dense_1_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_2_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_3_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_4_units, activation='relu'))
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
early_stopping = EarlyStopping(monitor='val_accuracy', patience=early_stopping, verbose=1, mode='auto', restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# model.fit_generator(train_generator, epochs=num_epochs, validation_data=validation_generator, callbacks=[early_stopping, learning_rate_callback])

# Fit the model
model.fit(
    train_generator.generate_data(),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator.generate_data(),
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[learning_rate_callback, checkpoint, custom_early_stopping, early_stopping],
)

# Save the model
model.save(f"{model_name}.keras")

