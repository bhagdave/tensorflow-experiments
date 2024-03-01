import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import random
import json
from SharedClasses import f1_score, CustomEarlyStopping, CustomImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import DepthwiseConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperModel
os.environ['TF_GRPC_TIMEOUT'] = '3600'  # Set it to 1 hour (3600 seconds)

image_folder = './images-new/close_up'
image_height = 256
image_width = 256
model_name = 'belron-simple-tuner'
batch_size = 8
num_classes = 2
num_epochs = 10

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Initialize the CustomImageDataGenerator for training and validation
train_generator = CustomImageDataGenerator(os.path.join(image_folder, 'train/'), image_width, image_height, batch_size=batch_size)
validation_generator = CustomImageDataGenerator(os.path.join(image_folder, 'validate/'), image_width, image_height, batch_size=batch_size)


class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()
        model.add(Conv2D(hp.Int('conv_1_units', min_value=32, max_value=256, step=32),(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # ... other layers ...
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
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate))
        model.add(Dense(hp.Int('dense_1_units', min_value=32, max_value=1024, step=32), activation='relu'))
        model.add(Dense(hp.Int('dense_2_units', min_value=32, max_value=1024, step=32), activation='relu'))
        model.add(Dense(hp.Int('dense_3_units', min_value=32, max_value=1024, step=32), activation='relu'))
        model.add(Dense(hp.Int('dense_4_units', min_value=32, max_value=1024, step=32), activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))


        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model


hypermodel = MyHyperModel(input_shape=(image_height, image_width, 3), num_classes=num_classes)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=5,
    directory='tuning',
    project_name='belron_simple_tuning'
)
checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_accuracy', save_best_only=True, mode='auto')
# Define the early stopping criteria
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


tuner.search(
    train_generator.generate_data(),
    epochs=num_epochs,
    validation_data=validation_generator.generate_data(),
    steps_per_epoch=train_generator.calculate_num_samples() // batch_size,
    validation_steps=validation_generator.calculate_num_samples() // batch_size,
    callbacks=[checkpoint, early_stopping, learning_rate_callback]
)

best_model = tuner.get_best_models(num_models=1)[0]
# Save the model
best_model.save(f"{model_name}-best.keras")

