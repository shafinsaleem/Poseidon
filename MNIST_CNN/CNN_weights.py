import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_parameters import CKKSParameters
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_bootstrapping_context import CKKSBootstrappingContext

# Initialize CKKS Parameters
poly_degree = 8192
ciph_modulus = 1 << 600
big_modulus = 1 << 1200
scaling_factor = 1 << 40
params = CKKSParameters(poly_degree=poly_degree,
                        ciph_modulus=ciph_modulus,
                        big_modulus=big_modulus,
                        scaling_factor=scaling_factor)

# Initialize the Key Generator and generate the keys
key_generator = CKKSKeyGenerator(params)
public_key = key_generator.public_key
secret_key = key_generator.secret_key
relin_key = key_generator.relin_key
conj_key = key_generator.generate_conj_key()

# Initialize Encoder, Encryptor, Decryptor, and Evaluator
encoder = CKKSEncoder(params)
encryptor = CKKSEncryptor(params, public_key, secret_key)
decryptor = CKKSDecryptor(params, secret_key)
evaluator = CKKSEvaluator(params)

# Initialize the Bootstrapping Context
bootstrapping_context = CKKSBootstrappingContext(params)

# Load the MNIST dataset and prepare
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0

# Define the CNN model (LeNet-5 like architecture)
def leNet_model():
    model = Sequential([
        Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(15, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = leNet_model()
print(model.summary())

# **Encrypt Model Weights**
def encrypt_weights(weights):
    """Encrypt the model weights using CKKS."""
    encrypted_weights = []
    for weight in weights:
        weight_flat = weight.flatten()
        encoded_weight = encoder.encode(weight_flat, scaling_factor)
        encrypted_weight = encryptor.encrypt(encoded_weight)
        encrypted_weights.append(encrypted_weight)
    return encrypted_weights

# **Decrypt Model Weights**
def decrypt_weights(encrypted_weights):
    """Decrypt the model weights using CKKS."""
    decrypted_weights = []
    for encrypted_weight in encrypted_weights:
        decrypted_weight = decryptor.decrypt(encrypted_weight)
        decoded_weight = np.array(encoder.decode(decrypted_weight)).real
        decrypted_weights.append(decoded_weight)
    return decrypted_weights

# Retrieve model weights and encrypt them
print("Encrypting model weights...")
model_weights = model.get_weights()
print("Model weights retrieved:", model_weights)
encrypted_weights = encrypt_weights(model_weights)
print("Model weights encrypted.")

# Example Homomorphic Operation (Optional)
print("Performing homomorphic operation on encrypted weights...")
# (You could add, multiply, etc. encrypted weights here)
# encrypted_weights[0] = evaluator.add(encrypted_weights[0], encrypted_weights[0])
print("Homomorphic operation complete.")

# Decrypt the weights to verify
print("Decrypting model weights for verification...")
decrypted_weights = decrypt_weights(encrypted_weights)
print("Decryption complete.")

# Reshape decrypted weights back to original shape
for i in range(len(model_weights)):
    decrypted_weights[i] = decrypted_weights[i].reshape(model_weights[i].shape)

# Assign decrypted weights back to the model and verify they match
model.set_weights(decrypted_weights)

# Training model on original dataset after weight decryption
print("Starting CNN model training...")
history = model.fit(X_train, y_train, epochs=1, batch_size=200, validation_data=(X_test, y_test), verbose=2)
print("CNN model training complete.")

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {score[0]} / Test accuracy: {score[1]}")
