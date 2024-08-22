import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import mnist

from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_parameters import CKKSParameters
from ckks.ckks_key_generator import CKKSKeyGenerator

# Function to pad data to the target size
def pad_data(data, target_size):
    padded_data = np.zeros(target_size)
    padded_data[:len(data)] = data
    return padded_data

# Load the MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Pick a single image and normalize
image = X_train[0].reshape(28, 28) / 255.0  # Normalize the image
image_flat = image.flatten()

# Define CKKS Parameters
poly_degree = 2048  # Adjust as needed
ciph_modulus = 1 << 40  # Adjust as needed
big_modulus = 1 << 120  # Adjust as needed
scaling_factor = 1 << 30  # Adjust as needed

# Initialize CKKS Parameters and Keys
params = CKKSParameters(poly_degree=poly_degree, ciph_modulus=ciph_modulus, big_modulus=big_modulus, scaling_factor=scaling_factor)
keygen = CKKSKeyGenerator(params)
public_key = keygen.public_key
secret_key = keygen.secret_key
relin_key = keygen.relin_key
conj_key = keygen.generate_conj_key()
print("generating rot keys")
rot_keys = {i: keygen.generate_rot_key(i) for i in range(1, poly_degree // 2)}  # Rotation keys
# Generate rotation keys for powers of 2
# rot_keys = {i: keygen.generate_rot_key(i) for i in [2**j for j in range(10)]}  # Powers of 2 up to 512

# Initialize Encoder, Encryptor, Decryptor, Evaluator
encoder = CKKSEncoder(params)
encryptor = CKKSEncryptor(params, public_key, secret_key)
decryptor = CKKSDecryptor(params, secret_key)
evaluator = CKKSEvaluator(params)
print("Initialization complete.")

# Pad the image to match the number of slots (target_size = poly_degree // 2)
target_size = poly_degree // 2
padded_image_flat = pad_data(image_flat, target_size)

print("Encrypting image...")
# Encode and encrypt the padded image
encoded_image = encoder.encode(padded_image_flat, scaling_factor)
encrypted_image = encryptor.encrypt(encoded_image)

# Decrypt the image before bootstrapping for comparison
decrypted_image_before_bootstrap = decryptor.decrypt(encrypted_image)
decoded_image_before_bootstrap = np.array(encoder.decode(decrypted_image_before_bootstrap)[:784]).real.reshape(28, 28)

# Apply bootstrapping
print("Applying bootstrapping...")
old_ciph, bootstrapped_image = evaluator.bootstrap(encrypted_image, rot_keys, conj_key, relin_key, encoder)
print("Bootstrapping complete.")

# Decrypt the image after bootstrapping
decrypted_image_after_bootstrap = decryptor.decrypt(bootstrapped_image)
decoded_image_after_bootstrap = np.array(encoder.decode(decrypted_image_after_bootstrap)[:784]).real.reshape(28, 28)

# Visualize original, decrypted before bootstrapping, and decrypted after bootstrapping
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

# Decrypted Image Before Bootstrapping
plt.subplot(1, 3, 2)
plt.imshow(decoded_image_before_bootstrap, cmap='gray')
plt.title("Decrypted Before Bootstrapping")

# Decrypted Image After Bootstrapping
plt.subplot(1, 3, 3)
plt.imshow(decoded_image_after_bootstrap, cmap='gray')
plt.title("Decrypted After Bootstrapping")

plt.show()
