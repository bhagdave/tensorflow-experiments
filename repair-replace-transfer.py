import tensorflow as tf
import tensorflow_hub as hub
import urllib.request
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)
pre_trained_model = InceptionV3(input_shape=(300, 300, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
x = Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)
model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])

image_folder = '/home/dave/Projects/tensorflow/tensorflow-experiments/images'
image_height = 300
image_width = 300
model_name = 'repair-replace-transfer'

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
            class_mode='binary'
)
test_datagen = ImageDataGenerator(
        rescale=1./255, 
)
test_generator = train_datagen.flow_from_directory(
            os.path.join(image_folder, 'validate/'),
            target_size=(image_width, image_height),
            batch_size=10,
            class_mode='binary'
)

model.fit_generator(train_generator, epochs=40, validation_data=test_generator)

model.save(model_name)

