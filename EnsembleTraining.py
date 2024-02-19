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

