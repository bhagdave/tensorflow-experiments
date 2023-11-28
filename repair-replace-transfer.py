import tensorflow as tf
import urllib.request
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image


image_folder = '/home/dave/Projects/tensorflow/tensorflow-experiments/images-new'
model_name = 'repair-replace-transfer'
image_height = 300
image_width = 300

# Define a custom data generator
class CustomImageDataGenerator:
    def __init__(self, directory, image_width, image_height, batch_size=32, class_mode='categorical'):
        self.directory = directory
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.image_files = self.collect_image_files()

    def collect_image_files(self):
        print("Debug: Collecting image files")
        image_files = {}

        for category in os.listdir(self.directory):
            category_dir = os.path.join(self.directory, category)
            if os.path.isdir(category_dir):
                image_files[category] = []
                for image_file in os.listdir(category_dir):
                    if image_file.endswith('.jpg'):
                        image_files[category].append(os.path.join(category_dir, image_file))

        return image_files

    def generate_data(self):
        while True:
            print ('DEBUG: Generating data')
            batch_images = []
            batch_labels = []

            for _ in range(self.batch_size):
                # Randomly select a category
                selected_category = np.random.choice(list(self.image_files.keys()))
                image_file = np.random.choice(self.image_files[selected_category])
                
                # Load and preprocess the selected image
                image = Image.open(image_file)
                image = image.resize((self.image_width, self.image_height))
                image = np.array(image) / 255.0  # Normalize the pixel values

                batch_images.append(image)
                batch_labels.append(selected_category)  # Use category as label

            batch_images = np.array(batch_images)
            if self.class_mode == 'categorical':
                # Convert category labels to one-hot encoding
                unique_categories = list(self.image_files.keys())
                batch_labels = [unique_categories.index(label) for label in batch_labels]
                batch_labels = tf.keras.utils.to_categorical(batch_labels, len(unique_categories))

            yield batch_images, batch_labels

# Use the custom data generator
print("Debug: Train Directory:", os.path.join(image_folder, 'train/'))
print("Debug: Validation Directory:", os.path.join(image_folder, 'validate/'))

train_datagen = CustomImageDataGenerator(
    directory=os.path.join(image_folder, 'train/'),
    image_width=image_width,
    image_height=image_height,
    batch_size=10,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

test_datagen = CustomImageDataGenerator(
    directory=os.path.join(image_folder, 'validate/'),
    image_width=image_width,
    image_height=image_height,
    batch_size=10,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

train_generator = train_datagen.generate_data()
test_generator = test_datagen.generate_data()

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
x = layers.Dense(4, activation='softmax')(x)
model = Model(pre_trained_model.input, x)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])


model.fit_generator(train_generator, epochs=40, validation_data=test_generator)

model.save(f"{model_name}.h5")

