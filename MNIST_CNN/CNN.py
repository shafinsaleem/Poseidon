# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical

# Import CKKS-related classes from pyFHE
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_parameters import CKKSParameters
from ckks.ckks_key_generator import CKKSKeyGenerator  # Your provided key generator

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

# Load the MNIST dataset and prepare as before
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0

# Flatten and pad the data
def pad_data(data, target_size):
    padded_data = np.zeros(target_size)
    padded_data[:len(data)] = data
    return padded_data

target_size = poly_degree // 2  # The target size should match the available slots in CKKS

# Visualize the original image
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title("Original Image")
plt.show()

# Encrypt the training data (example for one image)
X_train_encrypted = [
    encryptor.encrypt(encoder.encode(pad_data(img.flatten(), target_size), scaling_factor))
    for img in X_train[:1]
]

# Apply some homomorphic operations on the first ciphertext
X_train_encrypted[0] = evaluator.add(X_train_encrypted[0], X_train_encrypted[0])  # Example operation

# Determine matrix_len based on the bootstrapping matrices used (adjust if necessary)
matrix_len = poly_degree // 2  # Adjust this based on actual matrix dimensions

# Calculate matrix_len_factor1 and matrix_len_factor2
matrix_len_factor1 = int(np.sqrt(matrix_len))
if matrix_len != matrix_len_factor1 * matrix_len_factor1:
    matrix_len_factor1 = int(np.sqrt(2 * matrix_len))
matrix_len_factor2 = matrix_len // matrix_len_factor1

# Generate rotation keys for all required shifts (from 1 to matrix_len_factor1 - 1)
rotation_shifts = list(range(1, matrix_len_factor1)) + [matrix_len_factor1 * j for j in range(matrix_len_factor2)]
rot_keys = {shift: key_generator.generate_rot_key(rotation=shift) for shift in rotation_shifts}

# Perform bootstrapping manually using the available keys and evaluator functions
# Step 1: Coeff-to-slot transformation
ciph0, ciph1 = evaluator.coeff_to_slot(X_train_encrypted[0], rot_keys, conj_key, encoder)

# Step 2: Exponentiation (or other function) to reduce noise
ciph_exp0 = evaluator.exp(ciph0, const=1.0, relin_key=relin_key, encoder=encoder)
ciph_exp1 = evaluator.exp(ciph1, const=1.0, relin_key=relin_key, encoder=encoder)

# Step 3: Slot-to-coeff transformation to return to original form
X_train_encrypted[0] = evaluator.slot_to_coeff(ciph_exp0, ciph_exp1, rot_keys, encoder)

# Step 4: Rescale the ciphertext to reduce noise
X_train_encrypted[0] = evaluator.rescale(X_train_encrypted[0], scaling_factor)

# Step 5: Relinearize the ciphertext to maintain its dimensionality
X_train_encrypted[0] = evaluator.relinearize(relin_key, X_train_encrypted[0])

# Decrypting and decoding for visualization
X_train_decrypted = [
    np.array(encoder.decode(decryptor.decrypt(enc_img))[:784]).real.reshape(28, 28, 1)
    for enc_img in X_train_encrypted
]

# Visualize the decrypted image to check correctness
plt.imshow(X_train_decrypted[0].reshape(28, 28), cmap='gray')
plt.title("Decrypted Image After Bootstrapping")
plt.show()

# Define the CNN model (LeNet-5 like architecture)
def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = leNet_model()
print(model.summary())

# Training the model on decrypted data (as a placeholder)
history = model.fit(X_train_decrypted, y_train[:1], epochs=20, batch_size=200, validation_data=(X_test[:1], y_test[:1]), verbose=2)

# Evaluate the model
score = model.evaluate(X_test[:1], y_test[:1], verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
