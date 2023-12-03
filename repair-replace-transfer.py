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
from tensorflow.keras.callbacks import EarlyStopping


image_folder = '/home/dave/Projects/tensorflow/tensorflow-experiments/images-new'
model_name = 'repair-replace-transfer'
image_height = 300
image_width = 300

early_stopping = EarlyStopping(
    monitor='val_loss',  # You can use 'val_accuracy' or other metrics as well
    patience=2,           # Number of epochs with no improvement after which training will stop
    restore_best_weights=True  # Restores the model weights from the epoch with the best monitored value
)

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
    return logs.get('loss') < 0.2  # Example: Stop if loss is less than 0.2

# Create the custom callback
custom_early_stopping = CustomEarlyStopping(condition=custom_condition, verbose=1)

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
        image_files = {}

        for category in os.listdir(self.directory):
            category_dir = os.path.join(self.directory, category)
            if os.path.isdir(category_dir):
                image_files[category] = []
                for image_file in os.listdir(category_dir):
                    if image_file.endswith('.jpg'):
                        image_files[category].append(os.path.join(category_dir, image_file))

        return image_files

    def generate_data(self, is_training=True):
        categories = os.listdir(self.directory)  # List of category folder names
        all_cases = []  # Collect all cases in all categories

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
            batch_images = []
            batch_labels = []

            for (category, guid) in all_cases:
                for image_type in ['close_up', 'damage_area']:
                    image_file = f"{guid}_{image_type}.jpg"
                    image_path = os.path.join(self.directory, category, image_file)
                    image = Image.open(image_path)
                    image = image.resize((self.image_width, self.image_height))
                    image = np.array(image) / 255.0  # Normalize the pixel values

                    batch_images.append(image)
                    batch_labels.append(category) 

                    if len(batch_images) == self.batch_size:
                        batch_images = np.array(batch_images)
                        if self.class_mode == 'categorical':
                            # Convert image types to one-hot encoded labels
                            batch_labels = np.array(batch_labels)
                            unique_labels = np.unique(batch_labels)
                            label_to_index = {label: i for i, label in enumerate(unique_labels)}
                            batch_labels = [label_to_index[label] for label in batch_labels]
                            batch_labels = tf.keras.utils.to_categorical(batch_labels, len(unique_labels))

                        yield batch_images, batch_labels

                        # Clear the batch lists
                        batch_images = []
                        batch_labels = []


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
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])


model.fit_generator(train_generator, epochs=40, validation_data=test_generator, callbacks=[early_stopping, custom_early_stopping])

model.save(f"{model_name}.h5")

