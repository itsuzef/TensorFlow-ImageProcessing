import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Assigns the name fashion_mnist to a particular dataset located within Keras
# The second line defines four arrays containing the training and testing data
# cleaved again into separate structures for images and labels,
# The training data arrays will be used to train the model,
# and the testing arrays will allow us to evaluate the performance of our model.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# Visualize the data using matplot
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Preprocessing the dataset
train_images = train_images / 255.0
test_images = test_images / 255.0


# The first layer – the Input Layer
# Creates a Flatten layer that intakes a multi-dimensional array and produces an array of a single dimension.
# This places all the pixel data on an equal depth during input.
# Rectified Linear Unit (ReLU) Activation Function that outputs values between zero and 1
# The activation function behaves like f(x)=max(0,x)
# 128 nodes - dense layer
# 10 nodes - dense layer
# The final layer uses the softmax activation function
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(
    128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])

# Train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)
predictions[0]
numpy.argmax(predictions[0])
