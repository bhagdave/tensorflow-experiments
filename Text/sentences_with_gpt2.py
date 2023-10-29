import tensorflow as tf
import os  # Import the os library
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
from transformers import TFGPT2LMHeadModel, GPT2Config, GPT2Tokenizer

folder_path = '/home/dave/Projects/tensorflow-experiments/Text'  # Replace with the path to your folder



def custom_loss_with_debug(y_true, y_pred):
    # Debug statements to check the inputs
    print("Inside custom loss function")
    print("y_true shape:", tf.shape(y_true))
    print("y_pred shape:", tf.shape(y_pred))
    print("y_true dtype:", y_true.dtype)
    print("y_pred dtype:", y_pred.dtype)
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    # Cast y_true to float32
    y_true = tf.cast(y_true, dtype=tf.float32)

    # Reduce the y_pred tensor by taking the maximum along the last dimension (vocabulary size)
    # This is a simplification; you might need a more specific reduction depending on your use case
    y_pred_reduced = tf.reduce_max(y_pred, axis=-1)

    # Compute mean squared error loss
    mse = tf.reduce_mean(tf.square(y_true - y_pred_reduced))

    return mse




# Initialize GPT-2 tokenizer and model
config = GPT2Config.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel(config)

# Placeholder list to hold tokenized Slack messages
tokenized_messages = []

# Loop through .txt files (assuming each file is a collection of Slack messages)
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            messages = file.readlines()
        # Tokenize each message using GPT-2 tokenizer
        for message in messages:
            input_ids = tokenizer.encode(message)
            tokenized_messages.append(input_ids)

padded_messages = pad_sequences(tokenized_messages, padding='post')
data = tf.convert_to_tensor(padded_messages, dtype=tf.int32)
labels = tf.roll(data, -1, axis=-1)

print("Data shape: ", data.shape)
print("Labels shape: ", labels.shape)
print("Data Type: ", data.dtype)
print("Labels Type: ", labels.dtype)
# Convert TensorFlow tensors to NumPy arrays
data_np = data.numpy()
labels_np = labels.numpy()

# Check for None or NaN values in data and labels
contains_none_data = np.any(data_np == None)
contains_nan_data = np.isnan(data_np).any()

contains_none_labels = np.any(labels_np == None)
contains_nan_labels = np.isnan(labels_np).any()

print("Data contains None: ", contains_none_data)
print("Data contains NaN: ", contains_nan_data)
print("Labels contain None: ", contains_none_labels)
print("Labels contain NaN: ", contains_nan_labels)

# Train model
# Configure the model and optimizer
# Define the configuration if customising (optional)
config = GPT2Config.from_pretrained("gpt2")

# Reinitialize the model
model = TFGPT2LMHeadModel(config)
model.build(input_shape=(2, 16, config.n_embd))  # Replace 11 with your sequence length
sample_data = data[:1]  # Taking just the first sample from data for demonstration

# Perform a forward pass to get the model's predictions
with tf.device("CPU:0"):  # Explicitly run on CPU to avoid any GPU-related issues for this test
    model_outputs = model(sample_data)

# model_outputs will be a dictionary containing various tensors. 
# For GPT-2, the key 'logits' will contain the output logits
output_logits = model_outputs.logits

print("Output logits shape:", output_logits.shape)
print("Output logits:", output_logits)


model.summary()
# Compile the model again
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer)

# Train the model
model.fit(data, labels, epochs=3, batch_size=16)
