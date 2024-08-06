import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Function to save an MNIST image
def save_mnist_image(data, index, filename):
    img = Image.fromarray(data[index])
    img = img.convert("L")
    img.save(filename)

# Save a few sample images from the test set
for i in range(10):
    save_mnist_image(X_test, i, f"mnist_sample_{i}.png")

print("Sample images saved.")
