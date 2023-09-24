import tensorflow as tf
import os
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


image_folder = '/home/dave/Projects/tensorflow/tensorflow-experiments/images'
image_height = 300
image_width = 300


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

model_name = 'repair-replace-cross'
model_path = f"{model_name}.h5"
model = load_model(model_path, custom_objects={'f1_score': f1_score})

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', f1_score])

train_datagen = ImageDataGenerator(
        rescale=1./255, 
        horizontal_flip=True, 
        vertical_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        zoom_range=0.2
)
train_generator = train_datagen.flow_from_directory(
            os.path.join(image_folder, 'train/'),
            target_size=(image_width, image_height),
            batch_size=10,
            class_mode='categorical'
)
test_datagen = ImageDataGenerator(
        rescale=1./255, 
)
test_generator = test_datagen.flow_from_directory(
            os.path.join(image_folder, 'validate/'),
            target_size=(image_width, image_height),
            batch_size=10,
            class_mode='categorical'
)

model.fit(
        train_generator,
        epochs=45,
        steps_per_epoch=67,
        validation_data=test_generator,
        validation_steps=10
)

model.save(f"{model_name}.h5")
