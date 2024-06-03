import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import sklearn.linear_model
from tensorflow import keras
from time import time


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


x_train, y_train = load_sign_mnist(load_all=True, train=True)
x_train = x_train.reshape(len(x_train), 28*28) / 255

# The letter J is not part of the dataset, therefore there is no integer label 9.
# We account for that and decrease labels > 9 by one.
y_train = np.array([yi-1 if yi > 9 else yi for yi in y_train])
# One-hot Encoding of labels
y_train = keras.utils.to_categorical(y_train)

plt.imshow(x_train[0].reshape(28,28), cmap="gray")
plt.title("Buchstabe D")
plt.show()

print('This is how a one-hot encoded Label of the letter D (integer label 3) looks like')
print(y_train[0])

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

'''
Notation: 
θ... Weights you want to train
x... 2D array, a batch of n images with 28*28 pixel.  
     x.shape should return (n, 784).
     For example x_train[10:42] could be such an x.
y... One-hot encoded labels corresponding to x.
'''


def linear(θ,x):
    """
    Linear function applied to each image in x.
    If x.shape == (n, 784), then it returns an (n, 24) array.
    """
    if x.ndim==1:
        return θ[0] + x@θ[1:] # Apply linear function to x if it contains only one image.
    else:
        return np.array([θ[0] + img@θ[1:] for img in x]) # Apply linear function to each image.


def softmax(x):
    """
    Softmax function. Make sure the function works for both 1D and 2D array.
    Be careful and only take the softmax for each row of x seperately!
    """
    if x.ndim == 1:
        exps = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exps / np.sum(exps)
    else:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


def h(θ, x):
    # Return 2D array of "probabilities" with shape (x.shape[0], 24).
    # Use linear and softmax to construct your classifier
    return softmax(linear(θ, x))


def loss(θ, x, y):
    # Cross-Entropy loss.
    # DEINE ANTWORT HIER
    return -np.mean(np.sum(y*np.log(h(θ, x)), axis=1))


# Softmax function should return values in [0,1]
v = np.random.uniform(low=-10, high=10, size=100)
print('Softmax function should return values in [0,1]. We test this with 100 randomly generated values.')
assert np.all(0 <= softmax(v))
assert np.all(softmax(v) <= 1)
print('Success!')

v = np.random.uniform(low=-10, high=10, size=200).reshape(20,10)
print('The sum of each row after softmax should also equal 1.')
print('Does it work for a single row vector?')
assert np.isclose(np.sum(softmax(v[0])), 1, atol=1e-8)
print('Yes!')
print('Does it work for a 2D array?')
assert np.isclose(np.sum(softmax(v),axis=1), np.ones(20), atol=1e-8).all() # Are all values close to 1?
print('Yes!')
print('The softmax function passed all tests.')


def gradient(θ, x, y):
    # This is the analytical gradient
    # We use it to massively decrease the computation time
    n = x.shape[0]
    xbias = np.hstack((np.ones((n,1)),x))
    return xbias.T@(h(θ,x)-y)/n


indices = np.arange(len(y_train)) # Indices from 0 to total number of images.
np.random.shuffle(indices) # Shuffle indices

θ = np.zeros((1 + 28*28, 24)) # The weight for h.
lr = 0.15 # Learning rate, should work well for this example

'''
TRAINING - This might take some time! We train for 20 epochs!

We will split the data into 20 parts, by splitting the indices with np.split.
'''
print('Initial loss:', loss(θ, x_train, y_train))  # The loss should decrease over time
for e in range(20):  # We train for 20 epochs
    for idx in np.array_split(indices, 20):  # idx are roughly 1/20 of all indices.
        x, y = x_train[idx], y_train[idx]
        # Here you update θ with gradient descent inside this loop.
        # Use x, y and the learning rate lr.

        θ = θ - lr * gradient(θ, x, y)

    print(f'Epoch {e + 1:2d}: loss =', loss(θ, x_train, y_train))


def predict(x):
    '''
    Predict the class of x using the classifier h.
    '''
    if len(x.shape) == 1:
        return np.argmax(h(θ, x))
    else:
        return np.argmax(h(θ, x), axis=-1)


k = 11 # Choose a number to display the k-th image
plt.imshow(x_train[k].reshape((28,28)), cmap="binary") # Show k-th image
plt.suptitle(f'We predict this is a {LETTERS[predict(x_train[k])]}')
plt.title(f'Actual label: {LETTERS[np.argmax(y_train[k])]}')
plt.show()

# Tests for h
v = np.random.uniform(low=0, high=1, size=(20,28*28))
print('We perform some tests on h. We use 20 random images.')
print('Does h returns values in [0,1]?')
assert np.all(0 <= h(θ,v))
assert np.all(h(θ,v) <= 1)
print('Yes!')
print('We check if the sum of each output row is equal to 1.')
assert np.isclose(np.sum(h(θ,v), axis=1), np.ones(20), atol=1e-8).all() # Are all values close to 1?
print('This is the case!')

print('Now we check if the accuracy of your h is high enough.')
accuracy = sum(predict(x_train) == np.argmax(y_train, axis=1))/len(y_train)
print('If done correctly, your algorithm should easily achieve a 60% accuracy.')
print(f'{accuracy:.2%} of all training images get classified correctly.')
assert accuracy > 0.6 # Achieve at least 60% accuracy!
print('Success!')
print('')

