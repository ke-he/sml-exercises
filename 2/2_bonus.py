import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import sklearn.linear_model
from tensorflow import keras
from time import time


np.random.seed(42)

# Loads the sign mnist data
def load_sign_mnist(train = True, load_all = True):
    x_key = "x_"
    y_key = "y_"

    if train:
        x_key += "train"
        y_key += "train"
    else:
        x_key += "test"
        y_key += "test"

    with np.load("sign_mnist.npz") as f:
        images, labels = f[x_key], f[y_key]

    if not load_all:
        indices = np.isin(labels, [0, 1])
        images = images[indices]
        labels = labels[indices]

    return images, labels

# We load and normalize the training data
x_train, y_train = load_sign_mnist(load_all=True, train=True)
x_train = x_train.reshape(len(x_train), 28*28) / 255
# Because there is no "J"-image in the dataset, there is no label "9" and we modify all labels after "J"
y_train = np.array([yi-1 if yi > 9 else yi for yi in y_train])


begin = time() # Keep this here and do not change
# Use the LogisticRegression classifier from scikit.
# Name it h and keep almost all the default arguments.
# Exceptions: Use the 'saga' solver and tol=0.1.

h = sklearn.linear_model.LogisticRegression(solver='saga', tol=0.1, random_state=42, max_iter=1000)

h.fit(x_train, y_train)

# Dont move the code below and leave it unchanged
total_time = time()-begin
print(f'This block took {total_time:.2} seconds to execute.')
print('If this took longer than 15 seconds, then something went wrong.')
print("Make sure you actually used the 'saga' solver and chose tol=0.1")

print('Your classifier h should have the method predict.')
print('Therefore it should be possible to call h.predict(...).')
print('This should already work if you used LogisticRegression from sklearn.')

assert hasattr(h, 'predict')
print()
# How many training images get correctly classified
print('We test if you achieve at least 80% accuracy on the training data.')
accuracy = 100 * np.mean(h.predict(x_train) == y_train)
print(f'{accuracy:.2f}% of all training images get classified correctly.')
assert accuracy > 0.8
print()

print('We check if your solution took too much time to compute.')
print(f'total_time: {total_time:.2} seconds.')
assert total_time < 15

