import tensorflow as tf

# Load the original model
model = tf.keras.models.load_model('model_Mnist.h5')

# Save the model in a newer format
model.save('model_Mnist_new.h5', save_format='h5')
