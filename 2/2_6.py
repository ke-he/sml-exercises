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


LETTERS = ["A", "B"]

# Prepares sign mnist data for the exercise
x_train_ab, y_train_ab = load_sign_mnist(load_all=False, train=True)
x_train_ab = x_train_ab.reshape(len(x_train_ab), 28*28) / 255

print("Number of images:", y_train_ab.shape[0])
print("Images of letter A:", np.sum(np.isin(y_train_ab, [0])))
print("Images of letter B:", np.sum(np.isin(y_train_ab, [1])))

'''
θ... Weights you want to train
x... 2D array, a batch of n images with 28*28 pixel.  
     x.shape should return (n, 784).
'''


def linear(θ, x):
    '''
    Linear function applied to each image in x.
    If x.shape == (n, 784), then it returns an (n,) array.

    The symbol @ simply computes a matrix-vector multiplication.
    It's the same as np.matmul and often the same as np.dot.

    In our case we want a row vector, so we multiply from the left.
    '''
    return θ[0] + x @ θ[1:]


def g(z):
    '''
    Logistic function.
    Make sure that your function is vectorized.
    This means g(z) works for arrays and scalars.
    Look at last week's exercises!
    '''
    result = 1/(1 + np.exp(-z))

    # Do not change or remove the following code - necessary for stability
    result = np.clip(result, 0.001, 0.999)

    return result


def h(θ, x):
    '''
    Returns a vector of length x.shape[0]
    with probabilities that x is of class 0 (A).
    Use functions linear and g.
    '''
    return g(linear(θ, x))


def loss(θ, x, y):
    '''
    The cross-entropy loss function for two classes.
    '''
    h_θ = h(θ, x)
    return -np.mean(y * np.log(h_θ) + (1 - y) * np.log(1 - h_θ))


x_test_1 = np.array([[-1.0, 2.0, np.pi, 0.0, 42.0], [-1.0, 2.0, np.pi, 0.0, 42.0]])
θ_test_1 = np.linspace(0.0, 1.0, 6)
y_test_1 = np.array([np.exp(1), 123.0])
v = np.random.uniform(low=-500, high=500, size=100)

### Tests for logistic function
g_is = g(x_test_1)
g_soll = np.load("logistic_test_solution.npy")

if np.isclose(g_is, g_soll, atol=1e-10).all():
    print("The logistic function seems to be implemented correctly.")
else:
    print("Something went wrong with the logistic function.")

# Tests for logistic function
print('g(z) should return values within [0,1].')
assert np.all(0 <= g(v))
assert np.all(g(v) <= 1)
print('Success!')
print('g(0) should return 0.5.')
assert g(0) == 0.5
print('Success!')
print('')

### Tests for hypothesis
h_is = h(θ_test_1, x_test_1)
h_soll = np.load("classifier_test_solution.npy")

if np.isclose(h_is, h_soll, atol=1e-10).all():
    print("The classifier seems to be implemented correctly.")
else:
    print("Something went wrong with the classifier.")

### Tests for loss function
loss_is = loss(θ_test_1, x_test_1, y_test_1)
loss_soll = np.load("loss_test_solution.npy")

if np.isclose(loss_is, loss_soll, atol=1e-10).all():
    print("The loss function seems to be implemented correctly.")
else:
    print("Something went wrong with the loss function.")

plot_limits = 6.0
x_plot = np.linspace(-plot_limits, plot_limits, 500)
y_plot = g(x_plot)
plt.plot(x_plot, y_plot)
plt.title("Logistic Function")
plt.xlabel("$z$")
plt.ylabel("$g(z)$")
plt.show()

assert np.isclose(g_is, g_soll, atol=1e-10).all()
assert np.isclose(h_is, h_soll, atol=1e-10).all()
assert np.isclose(loss_is, loss_soll, atol=1e-10).all()


"""
Calculates the gradient of the loss function
"""
def gradient(θ, x, y):
    dθ = np.empty_like(θ) # Uninitialized array
    ϵ = 1e-4
    for k in range(len(θ)):
        e = np.zeros(θ.shape)
        e[k] = ϵ
        dθ[k] = loss(θ + e, x, y) - loss(θ - e, x, y)
    return dθ/(2*ϵ)


"""
Training

We will split the data into 20 parts, by splitting the indices with np.split.
"""

indices = np.arange(len(y_train_ab))
np.random.shuffle(indices)

θ = np.zeros(1 + 28**2)     # Weights for h
lr = 0.03                   # Learning Rate

# Complete the training loop using gradient descent
print("Initial loss: ", loss(θ, x_train_ab, y_train_ab))
for idx in np.array_split(indices, 40):
    image, label = x_train_ab[idx], y_train_ab[idx]

    # Gradient descent has to be implemented here.
    # Use image, label and the learning rate

    θ = θ - lr * gradient(θ, image, label)

    print("Loss: ", loss(θ, x_train_ab, y_train_ab))

decision_boundary = 0.5


def predict(x):
    '''
    Predict the class of x using the classifier h.
    0 => letter A
    1 => letter B
    '''
    p = h(θ, x)

    if isinstance(p, np.ndarray):
        larger = np.where(p >= decision_boundary)
        smaller = np.where(p < decision_boundary)

        p[larger] = 1
        p[smaller] = 0
    else:
        p = 1 if p >= decision_boundary else 0

    return p

x_test_ab, y_test_ab = load_sign_mnist(load_all=False, train=False)
x_test_ab = x_test_ab.reshape(len(x_test_ab), 28*28) / 255

y_pred_ab = predict(x_test_ab)

accuracy = np.sum(y_pred_ab == y_test_ab) / len(y_test_ab)

indices = np.arange(len(y_test_ab))
np.random.shuffle(indices)
indices = indices[:100]

x_plot_ab = x_test_ab[indices]
y_plot_ab = y_test_ab[indices]

A_indices = np.where(y_plot_ab == 0)
B_indices = np.where(y_plot_ab == 1)

x_plot_a = x_plot_ab[A_indices]
x_plot_b = x_plot_ab[B_indices]

linears = linear(θ, x_plot_ab)
offset = 0.5

bounds_max = np.max(linears) + offset
bounds_min = -bounds_max

x_line = np.linspace(bounds_min, bounds_max)
y_line = np.zeros(len(x_line)) + decision_boundary

plt.plot(x_line, y_line, linestyle="dotted", color="lightgray", linewidth=0.8)
plt.plot(x_line, g(x_line), linestyle="dashed", color="black", linewidth=0.8, label="logistic")
plt.plot()

plt.fill_between(x_line, decision_boundary, 1.05, color='royalblue', alpha=0.1)
plt.fill_between(x_line, -0.05, decision_boundary, color='firebrick', alpha=0.1)

plt.scatter(linear(θ, x_plot_a), h(θ, x_plot_a), marker="x", color="firebrick", label="A")
plt.scatter(linear(θ, x_plot_b), h(θ, x_plot_b), marker="x", color="royalblue", label="B")

plt.xlim(bounds_min, bounds_max)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.title("Visualization of some test data")

plt.ylabel("$h_\\theta(x)$")
plt.xlabel("$\\theta^T\\,x$")
plt.show()

print(f"Model accuracy is {accuracy*100:.2f}%!")

# Diese Zelle kann verwendet werden um das Modell auf zufälligen Bildern (A oder B) zu testen
random_index = np.random.choice(np.arange(len(y_test_ab)))
random_image = x_test_ab[random_index]
random_label = LETTERS[y_test_ab[random_index]]

prediction = LETTERS[predict(random_image)]

if prediction == random_label:
    title = f"The model correctly classified this image as {prediction}."
else:
    title = f"The model claims this is a(n) {prediction}, actually it is a(n) {random_label}."

plt.imshow(random_image.reshape(28, 28), cmap="gray")
plt.axis("off")
plt.title(title)
plt.show()

x_train_ab, y_train_ab = load_sign_mnist(load_all=False, train=True)
x_train_ab = x_train_ab.reshape(len(x_train_ab), 28*28) / 255
print('Is the output of h plausible?')
print('Does the first image get correctly classified?')
p = h(θ, x_train_ab[0])

print('"Probability" that the first image is in the first class:', f"{1-p:.3%}")
assert 0 <= p <= 1
assert predict(x_train_ab[0]) == y_train_ab[0] # Does the first image get correctly classified?
print('Correctly classified!')

print('Does the second image get correctly classified?')
p = h(θ, x_train_ab[1])
print('"Probability" that the second image is in the second class:', f"{p:.3%}")
assert 0 <= p <= 1
assert predict(x_train_ab[1]) == y_train_ab[1] # Does the second image get correctly classified?
print('Correctly classified!')
print('')

print('Now we check if the accuracy of your h is high enough.')
accuracy = np.sum(predict(x_train_ab) == y_train_ab)/len(y_train_ab)
print('If done correctly, your algorithm should easily achieve a 90% accuracy.')
print(f'{accuracy:.2%} of all train images get classified correctly.')
assert accuracy > 0.9 # Achieve at least 90% accuracy!
print('Success!')
print('')
