from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
from SharedClasses import CustomImageDataGenerator, f1_score

model1_name='models/belron-simple-256.keras'
model2_name='models/repair-replace-resnet-cross.keras'
model3_name='models/repair-replace-cross-256.keras'
model4_name='models/belron-mobilenet.keras'


def load_ensemble_models(model_filenames):
    models = []
    for filename in model_filenames:
        models.append(load_model(filename, custom_objects={'f1_score': f1_score}))
    return models



def ensemble_predictions(models, generator, total_images):
    processed_images = 0

    for X_batch, _ in generator:
        batch_preds = [model.predict(X_batch) for model in models]
        avg_batch_pred = np.mean(batch_preds, axis=0)  # Average predictions from all models

        # Process each prediction in the batch
        for i, pred in enumerate(avg_batch_pred):
            # pred contains the prediction values (probabilities) for each class
            predicted_class = np.argmax(pred)
            print(f'Processed image {processed_images + 1}/{total_images}: Predicted class = {predicted_class}, Prediction values = {pred}')
            processed_images += 1

            if processed_images >= total_images:
                break

        if processed_images >= total_images:
            break



model_filenames = [model1_name, model2_name, model3_name, model4_name]
models = load_ensemble_models(model_filenames)

image_directory = 'images-testing/close_up'
image_width = 256 
image_height = 256
batch_size = 4   

data_generator = CustomImageDataGenerator(image_directory, image_width, image_height, batch_size)
generator = data_generator.generate_data(is_training=False)

ensemble_pred = ensemble_predictions(models, generator, data_generator.calculate_num_samples())

ensemble_labels = np.argmax(ensemble_pred, axis=1)
y_test_labels = np.argmax(['repair','replace'], axis=1) 

ensemble_accuracy = accuracy_score(y_test_labels, ensemble_labels)
print(f'Ensemble Model Accuracy: {ensemble_accuracy}')

