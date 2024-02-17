from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score

model1_name='model1.h5'
model2_name='model2.h5'
model3_name='model3.h5'


def load_ensemble_models(model_filenames):
    models = []
    for filename in model_filenames:
        models.append(load_model(filename))
    return models


def ensemble_predictions(models, generator):
    # Initialize an empty list to store predictions
    ensemble_preds = []

    # Iterate over the generator
    for X_batch, _ in generator:
        # Get predictions from each model for the current batch and average them
        batch_preds = [model.predict(X_batch) for model in models]
        avg_batch_pred = np.mean(batch_preds, axis=0)

        # Store the average predictions
        ensemble_preds.append(avg_batch_pred)

    # Concatenate all batch predictions
    final_predictions = np.concatenate(ensemble_preds, axis=0)
    return final_predictions


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

# Load models
model_filenames = [model1_name, model2_name, model3_name]
models = load_ensemble_models(model_filenames)

# Create an instance of your data generator
image_directory = 'path/to/your/image/directory'
image_width = 300  # Example width, adjust as needed
image_height = 300 # Example height, adjust as needed
batch_size = 8   # Example batch size, adjust as needed

data_generator = CustomImageDataGenerator(image_directory, image_width, image_height, batch_size)
generator = data_generator.generate_data(is_training=False)

# Generate ensemble predictions
ensemble_pred = ensemble_predictions(models, generator)

ensemble_labels = np.argmax(ensemble_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)  # Adjust if your y_test is not one-hot encoded

# Evaluate accuracy
ensemble_accuracy = accuracy_score(y_test_labels, ensemble_labels)
print(f'Ensemble Model Accuracy: {ensemble_accuracy}')

