from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os



def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

model_name = "belron-simple"
categories = ['repair', 'replace']
image_width = 300
image_height = 300
# Load the saved model
model = load_model(f"./models/{model_name}.keras", custom_objects={'f1_score': f1_score})

def generate_data_for_prediction(directory, image_width, image_height, batch_size):
    categories = os.listdir(directory)  # List of category folder names
    all_cases = []  # Collect all cases in all categories

    for category in categories:
        category_path = os.path.join(directory, category)
        image_files = os.listdir(category_path)
        guids = set()

        for image_file in image_files:
            if image_file.endswith('.jpg'):
                guid, _ = image_file.split('_')[:2]
                guids.add(guid)

        all_cases.extend([(category, guid) for guid in guids])

    while True:
        batch_images = []
        for _ in range(batch_size):
            if len(all_cases) == 0:
                break  # Break if no more images to process

            category, guid = all_cases.pop(0)
            combined_image = None
            for image_type in ['close_up', 'damage_area']:
                image_file = f"{guid}_{image_type}.jpg"
                image_path = os.path.join(directory, category, image_file)
                image = Image.open(image_path)
                image = image.resize((image_width, image_height))
                image_array = np.array(image) / 255.0

                if combined_image is None:
                    combined_image = image_array
                else:
                    combined_image = np.concatenate((combined_image, image_array), axis=-1)  # Combine along the channel axis

            batch_images.append(combined_image)

        if len(batch_images) == 0:
            break

        yield np.array(batch_images)  # Convert the list of images to a NumPy array


prediction_generator = generate_data_for_prediction(
    directory='./images-for-prediction/train',  # Path to the folder with images for prediction
    image_width=300,
    image_height=300,
    batch_size=8
)

# Use the generator to get predictions
for batch in prediction_generator:
    predictions = model.predict(batch)
    # Process predictions
    print("Predictions:", predictions)
    predicted_index = np.argmax(predictions, axis=1)
    print("Predicted Index:", predicted_index)
    predicted_category = [categories[i] for i in predicted_index]
    print("Predicted Category:", predicted_category)
