import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Add, Activation, Input, Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import SGD



image_folder = './images-for-prediction'
image_height = 256
image_width = 256
model_name = 'repair-replace-cross-256'
batch_size = 8
num_classes = 2
learning_rate = 0.001
dropout_rate1 = 0.5
dropout_rate2 = 0.4
regularisation_rate = 0.0002
early_stopping_patience = 10
num_epochs = 100


# Initialize the CustomImageDataGenerator for training and validation
train_generator = CustomImageDataGenerator(os.path.join(image_folder, 'train/'), image_width, image_height, batch_size=batch_size)
validation_generator = CustomImageDataGenerator(os.path.join(image_folder, 'validate/'), image_width, image_height, batch_size=batch_size)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(image_height, image_width, 3))

# First: train only the top layers (which were randomly initialized)
#for layer in base_model.layers:
#    layer.trainable = False

# Create the model
input_tensor = Input(shape=(image_height, image_width, 3))
x = base_model(input_tensor)
x = Dropout(dropout_rate1)(x)  # Apply dropout
x = BatchNormalization()(x)  # Batch normalization before activation

x = Flatten()(x)  # Flatten the output
x = Dropout(dropout_rate2)(x)  # Apply dropout again
predictions = Dense(num_classes, activation='softmax')(x)  # Final layer with softmax activation for classification

model = Model(inputs=input_tensor, outputs=predictions)

rmsprop_optimizer = Adam(learning_rate=learning_rate)
#rmsprop_optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True, decay=1e-6)


def scheduler(epoch, lr):
    if epoch < 10:
        return learning_rate
    elif epoch < 20:
        return learning_rate * .1
    elif epoch < 40:
        return learning_rate * .01
    else:
        return learning_rate * .001

model.compile(optimizer=rmsprop_optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_score])

# Reduce learning rate when a metric has stopped improving
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=8, min_lr=0.00005)

checkpoint = ModelCheckpoint('model-{epoch:03d}.keras', monitor='val_accuracy', save_best_only=True, mode='auto')
# Define the early stopping criteria
early_stopping_loss = EarlyStopping(monitor='val_loss',verbose=1, patience=early_stopping_patience, mode='min')
#early_stopping_accuracy = EarlyStopping(monitor='val_accuracy', min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='max')
#early_stopping_f1 = EarlyStopping(monitor='val_f1_score',min_delta=0.001,verbose=1, patience=early_stopping_patience, mode='max')
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.summary()
model.fit(
    x=train_generator.generate_data(),
    epochs=num_epochs,
    steps_per_epoch=train_generator.calculate_num_samples() // train_generator.batch_size,
    validation_data=validation_generator.generate_data(),
    validation_steps=validation_generator.calculate_num_samples() // validation_generator.batch_size,
    verbose=1,
    callbacks=[early_stopping_loss, checkpoint, learning_rate_callback],
)

# Save the model
print(f"Saving {model_name}")
model.save(f"models/{model_name}.keras")

