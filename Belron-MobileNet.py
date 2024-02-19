import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import random
import json
from SharedClasses import f1_score, CustomEarlyStopping, CustomImageDataGenerator
from keras.models import Sequential
from SharedClasses import f1_score, CustomEarlyStopping, CustomImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import DepthwiseConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Input


image_folder = './images-new/close_up'
image_height = 256
image_width = 256
model_name = 'belron-mobilenet'
batch_size = 8
num_classes = 2
learning_rate = 0.001
dropout_rate1 = 0.15
dropout_rate2 = 0.1
regularisation_rate = 0.03
early_stopping_patience = 40
num_epochs = 80
dense_layer_size = 512


def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Initialize the CustomImageDataGenerator for training and validation
train_generator = CustomImageDataGenerator(os.path.join(image_folder, 'train/'), image_width, image_height, batch_size=batch_size)
validation_generator = CustomImageDataGenerator(os.path.join(image_folder, 'validate/'), image_width, image_height, batch_size=batch_size)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

input_tensor = Input(shape=(image_height, image_width, 6))
x = Conv2D(3, (1, 1))(input_tensor)  # 1x1 convolution
x = base_model(x)
x = GlobalAveragePooling2D()(x)

# Add a Dropout layer after VGG16 model
x = Dropout(dropout_rate1)(x)  # 50% dropout

# Add a new top layer with L2 regularisation
x = Dense(dense_layer_size, activation='relu', kernel_regularizer=regularizers.l2(regularisation_rate))(x)
x = Dropout(dropout_rate2)(x)  # 50% dropout after the first Dense layer
predictions = Dense(num_classes, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=input_tensor, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

rmsprop_optimizer = RMSprop(learning_rate=learning_rate)

model.compile(optimizer=rmsprop_optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_score])

checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_accuracy', save_best_only=True, mode='auto')
# Define the early stopping criteria
early_stopping_loss = EarlyStopping(monitor='val_loss', min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='min')
early_stopping_accuracy = EarlyStopping(monitor='val_accuracy', min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='max')
early_stopping_f1 = EarlyStopping(monitor='val_f1_score',min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='max')
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.fit(
    x=train_generator.generate_data(),
    epochs=num_epochs,
    steps_per_epoch=train_generator.calculate_num_samples() // train_generator.batch_size,
    validation_data=validation_generator.generate_data(),
    validation_steps=validation_generator.calculate_num_samples() // validation_generator.batch_size,
    verbose=1,
    callbacks=[early_stopping_f1, early_stopping_accuracy, early_stopping_loss, checkpoint, learning_rate_callback]
)

# Save the model
model.save(f"{model_name}.keras")

