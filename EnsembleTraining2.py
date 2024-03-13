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



def ensemble_predictions(models, x_batch, batch_guids, processed_images, total_images):
    batch_preds = np.array([model.predict(x_batch) for model in models])
    avg_batch_pred = np.mean(batch_preds, axis=0)  # Average predictions from all models

    for i, pred in enumerate(avg_batch_pred):
        predicted_class = np.argmax(pred)
        guid = batch_guids[i]
        print(f'Guid:{guid} Predicted class = {predicted_class}, Prediction values = {pred}')
        processed_images += 1
        print(f'Processed image {processed_images + 1}/{total_images}')
        if processed_images >= total_images:
            break

    print(avg_batch_pred.shape)  # Should print (batch_size, number_of_classes)
    return avg_batch_pred, processed_images




model_filenames = [model1_name, model2_name, model3_name, model4_name]
models = load_ensemble_models(model_filenames)

image_directory = 'images-testing/close_up'
image_width = 256
image_height = 256
batch_size = 4

data_generator = CustomImageDataGenerator(image_directory, image_width, image_height, batch_size)

# Assuming generator is already defined and ensemble_predictions function is ready
generator = data_generator.generate_data(is_training=False)
total_images = data_generator.calculate_num_samples()  # Adjust based on your generator's method

all_true_labels = []
all_predicted_labels = []

processed_images = 0
for batch_images, batch_labels, batch_guids in generator:
    # Assuming ensemble_predictions function processes one batch at a time and returns predictions
    batch_predictions = ensemble_predictions(models, batch_images, batch_guids, processed_images, total_images)  # Adjust this call as necessary
    if processed_images >= total_images:
        break

