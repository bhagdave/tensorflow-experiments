import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras import regularizers


image_folder = './images-new/close_up'
image_height = 224
image_width = 224
model_name = 'repair-replace-cross'
batch_size = 8
num_classes = 2
learning_rate = 0.001
dropout_rate1 = 0.15
dropout_rate2 = 0.135
regularisation_rate = 0.0001
early_stopping_patience = 10
num_epochs = 100
dense_layer_size = 1024


# Initialize the CustomImageDataGenerator for training and validation
train_generator = CustomImageDataGenerator(os.path.join(image_folder, 'train/'), image_width, image_height, batch_size=batch_size)
validation_generator = CustomImageDataGenerator(os.path.join(image_folder, 'validate/'), image_width, image_height, batch_size=batch_size)

rmsprop_optimizer = Adam(learning_rate=learning_rate)

checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_loss', save_best_only=True, mode='auto')
# Define the early stopping criteria
early_stopping_loss = EarlyStopping(monitor='val_loss', min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='min')
early_stopping_accuracy = EarlyStopping(monitor='val_accuracy', min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='max')
early_stopping_f1 = EarlyStopping(monitor='val_f1_score',min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='max')

model = load_model(f"models/{model_name}.keras", custom_objects={'f1_score': f1_score})
model.compile(optimizer=rmsprop_optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_score])

model.fit(
    x=train_generator.generate_data(),
    epochs=num_epochs,
    steps_per_epoch=train_generator.calculate_num_samples() // train_generator.batch_size,
    validation_data=validation_generator.generate_data(),
    validation_steps=validation_generator.calculate_num_samples() // validation_generator.batch_size,
    verbose=1,
    callbacks=[early_stopping_f1, early_stopping_accuracy, early_stopping_loss, checkpoint]
)

# Save the model
model.save(f"{model_name}.keras")

