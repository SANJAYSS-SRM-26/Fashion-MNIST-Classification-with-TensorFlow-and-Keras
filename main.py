# day2
import tensorflow as tf
import matplotlib.pyplot as plt

# Check TensorFlow version
print(tf.__version__)

# Load the Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Display the first training image and its label
plt.imshow(training_images[0])
print("Label:", training_labels[0])
print("Image Data:")
print(training_images[0])

# Normalize the pixel values to be between 0 and 1
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model with an optimizer, loss function, and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data for 5 epochs
model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on the test data
model.evaluate(test_images, test_labels)

# Make predictions on the test images
classifications = model.predict(test_images)

# Display the classification probabilities for the first test image
print("Classification Probabilities for the First Test Image:")
print(classifications[0])





