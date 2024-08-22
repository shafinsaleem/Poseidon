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
poly_degree = 2048
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

# Flatten and pad the data
def pad_data(data, target_size):
    padded_data = np.zeros(target_size)
    padded_data[:len(data)] = data
    return padded_data

target_size = poly_degree // 2

# Encrypt the training data (example for one image)
print("Encrypting training data...")
X_train_encrypted = [
    encryptor.encrypt(encoder.encode(pad_data(img.flatten(), target_size), scaling_factor))
    for img in X_train[:1]
]
print("Encryption complete.")

# Apply some homomorphic operations
print("Applying homomorphic operations...")
X_train_encrypted[0] = evaluator.add(X_train_encrypted[0], X_train_encrypted[0])
print("Homomorphic operations complete.")

# Perform bootstrapping
print("Performing bootstrapping...")
matrix_len = poly_degree // 2
matrix_len_factor1 = int(np.sqrt(matrix_len))
if matrix_len != matrix_len_factor1 * matrix_len_factor1:
    matrix_len_factor1 = int(np.sqrt(2 * matrix_len))
matrix_len_factor2 = matrix_len // matrix_len_factor1
print("Matrix length factors calculated.")
rotation_shifts = list(range(1, matrix_len_factor1)) + [matrix_len_factor1 * j for j in range(matrix_len_factor2)]
print("Rotation shifts generated.")
rot_keys = {shift: key_generator.generate_rot_key(rotation=shift) for shift in rotation_shifts}
print("Rotation keys generated.")
ciph0, ciph1 = evaluator.coeff_to_slot(X_train_encrypted[0], rot_keys, conj_key, encoder)
print("Coefficient to Slot transformation complete.")

# Use bootstrapping context matrices
s1 = evaluator.multiply_matrix(ciph0, bootstrapping_context.encoding_mat_conj_transpose0, rot_keys, encoder)
s2 = evaluator.multiply_matrix(ciph1, bootstrapping_context.encoding_mat_conj_transpose1, rot_keys, encoder)
print("Matrix multiplication with bootstrapping context complete.")

ciph_exp0 = evaluator.exp(s1, const=1.0, relin_key=relin_key, encoder=encoder)
ciph_exp1 = evaluator.exp(s2, const=1.0, relin_key=relin_key, encoder=encoder)
print("Exponentiation complete.")

X_train_encrypted[0] = evaluator.slot_to_coeff(ciph_exp0, ciph_exp1, rot_keys, encoder)
X_train_encrypted[0] = evaluator.multiply_matrix(X_train_encrypted[0], bootstrapping_context.encoding_mat0, rot_keys, encoder)
X_train_encrypted[0] = evaluator.rescale(X_train_encrypted[0], scaling_factor)
X_train_encrypted[0] = evaluator.relinearize(relin_key, X_train_encrypted[0])
print("Bootstrapping complete.")

# Decrypting and decoding for visualization
print("Decrypting data for visualization...")
X_train_decrypted = [
    np.array(encoder.decode(decryptor.decrypt(enc_img))[:784]).real.reshape(28, 28, 1)
    for enc_img in X_train_encrypted
]
print("Decryption complete.")

# Visualize the decrypted image
plt.imshow(X_train_decrypted[0].reshape(28, 28), cmap='gray')
plt.title("Decrypted Image After Bootstrapping")
plt.show()

# CNN Model Training
print("Starting CNN model training...")
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
history = model.fit(X_train_decrypted, y_train[:1], epochs=20, batch_size=200, validation_data=(X_test[:1], y_test[:1]), verbose=2)
print("CNN model training complete.")

# Evaluate the model
score = model.evaluate(X_test[:1], y_test[:1], verbose=0)
print(f"Test loss: {score[0]} / Test accuracy: {score[1]}")
