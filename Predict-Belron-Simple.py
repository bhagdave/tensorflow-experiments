#!/usr/bin/env python3

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import csv
import argparse
from SharedClasses import f1_score

parser = argparse.ArgumentParser(description='Load a model and use it for prediction.')
parser.add_argument('model_name', type=str, help='Nname of the model to load and use for predictions.')

# Parse the command-line arguments
args = parser.parse_args()

# Extract the model filename
model_name = args.model_name


categories = ['repair', 'replace']
image_width = 256
image_height = 256
# Load the saved model
model = load_model(f"models/{model_name}.keras", custom_objects={'f1_score': f1_score})

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
        batch_filenames = []
        batch_categories = []

        for _ in range(batch_size):
            if len(all_cases) == 0:
                break  # Break if no more images to process

            category, guid = all_cases.pop(0)
            combined_image = None
            image_file = f"{guid}_close_up.jpg"
            image_path = os.path.join(directory, category, image_file)
            image = Image.open(image_path)
            image = image.resize((image_width, image_height))
            image_array = np.array(image) / 255.0

            batch_images.append(image_array)
            batch_filenames.append(image_file)
            batch_categories.append(category)

        if len(batch_images) == 0:
            break

        yield np.array(batch_images), batch_filenames, batch_categories  # Convert the list of images to a NumPy array

correct = 0
total = 0

prediction_generator = generate_data_for_prediction(
    directory='./images-unseen/close_up',  # Path to the folder with images for prediction
    image_width=image_width,
    image_height=image_height,
    batch_size=8
)

with open(f"{model_name}-predictions.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Human Category', 'Predicted Category', 'Confidence'])

    # Use the generator to get predictions
    for batch, filenames, human_categories in prediction_generator:
        predictions = model.predict(batch, verbose=0)
        # Process predictions
        predicted_index = np.argmax(predictions, axis=1)
        predicted_category = [categories[i] for i in predicted_index]
        confidence_scores = np.max(predictions, axis=1)

        for filename, human_category, category, score in zip(filenames, human_categories, predicted_category, confidence_scores):
            #print(f"Filename: {filename}, Human Assigned Category: {human_category}, Predicted Category: {category}, Confidence Score: {score}")
            writer.writerow([filename, human_category, category, score])
            total += 1
            if human_category == category:
                correct += 1
            
percentage = correct / total * 100
print(f"Correct: {correct}, Total: {total}, Percentage: {percentage:.2f}%")
