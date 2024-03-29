import tensorflow as tf
import json
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
os.environ['TF_GRPC_TIMEOUT'] = '3600'  # Set it to 1 hour (3600 seconds)

image_height = 300
image_width = 300
model_name = 'simple-repair-replace-cross'
image_folder = './images-new'
batch_size = 8
num_classes = 4

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
                combined_image= None
                for image_type in ['close_up', 'damage_area']:
                    image_file = f"{guid}_{image_type}.jpg"
                    image_path = os.path.join(self.directory, category, image_file)
                    image = Image.open(image_path)
                    image = image.resize((self.image_width, self.image_height))
                    image = np.array(image) / 255.0  # Normalize the pixel values
                    
                    if combined_image is None:
                        combined_image = image
                    else:
                        combined_image = np.concatenate((combined_image, image), axis=-1)  # Combine along the channel axis

                batch_labels.append(category)
                batch_images.append(combined_image)
                # Append the one-hot encoded label based on the category

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
    model.add(Conv2D(112, (3,3), activation='relu', input_shape=(image_height, image_width, 6)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(192, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(96, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(160, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(96, (3,3), activation='relu') )
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(112, activation='relu'))
    model.add(Dense(240, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model

model = model_builder()
model.summary()
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy', f1_score]
)

print("Training model")
# Fit the model
model.fit(
    train_generator.generate_data(),
    epochs=4,
    steps_per_epoch=100,
    validation_data=validation_generator.generate_data(),
    validation_steps=10,
    verbose=1
)

# Save the model
model.save(f"{model_name}.h5")

