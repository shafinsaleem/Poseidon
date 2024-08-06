import argparse
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('../model_Mnist.h5')

def predict_digit(img_path):
    # Open an image file
    with Image.open(img_path) as img:
        # resize image to 28x28 pixels
        img = img.resize((28, 28))
        # convert rgb to grayscale
        img = img.convert('L')
        img = np.array(img)
        # reshaping to support our model input and normalizing
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        # predicting the class
        res = model.predict([img])[0]
        return np.argmax(res), max(res)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict digit from an image using a trained model.')
    parser.add_argument('img_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    # Predict the digit
    digit, confidence = predict_digit(args.img_path)
    print(f'Predicted digit: {digit}, Confidence: {confidence*100:.2f}%')

if __name__ == "__main__":
    main()
