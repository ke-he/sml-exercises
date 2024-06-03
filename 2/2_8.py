import numpy as np
import matplotlib.pyplot as plt
import scipy

# Loading training and test data
with np.load('Aufgabe8.npz') as f:
    x_train, y_train = f['x_train'], f['y_train'],
    x_test, y_test = f['x_test'], f['y_test']

print(x_train.shape, y_train.shape)

plt.plot(x_train, y_train, "o", label="training data")
plt.plot(x_test, y_test, "o", label="test data")
plt.legend()
plt.grid(alpha=0.4)

degree = 11  # Maximal polynomial degree


def h(θ, x):
    # we switch the order of the thetas, as np.polyval expects the highest degree first
    return np.polyval(θ[::-1], x)


X = np.array([[x ** k for k in range(degree + 1)] for x in x_train])  # X matrix
print(X.shape)


def optimalθ(λ):
    # Returns the optimal θ for a given regularization parameter λ
    A = X.T @ X + λ * np.eye(len(x_train))  # np.eye is the identity matrix
    b = X.T @ y_train
    # Solve the linear system
    θ = scipy.linalg.solve(A, b)  # Solves Aθ = b
    return θ


neθ = optimalθ(0)  # Solution without regularization

plt.plot(x_train, y_train, "o", label="training data")
plt.plot(x_test, y_test, "o", label="test data")
#plt.plot(x_plot, h(neθ, x_plot), color="r", label="horrible overfit")
plt.grid(alpha=0.4)
plt.legend()
plt.show()


def loss(θ, x, y, λ):
    # The usual mean squared error loss, but with a regularization parameter
    m = len(y)
    mse_loss = np.mean((h(θ, x) - y) ** 2) / 2
    regularization = (λ / (2 * m)) * np.sum(θ ** 2)
    return mse_loss + regularization


# HERE YOU CHOOSE YOUR REGULARIZATION PARAMETER
λ = 0.56  # Ersetze mit deinem Parameter
# DEINE ANTWORT HIER
# Clarification: In this cell you only have to modify λ and and implement the loss, nothing else.

θ = optimalθ(λ)  # θ given for the regularization parameter λ you chose above

print(f"Loss auf den Testdaten: {loss(θ, x_test, y_test, λ): .4f}")
plt.clf()
x_plot = np.linspace(0,5,300)
# train and test data
plt.plot(x_train,y_train,"o",label="training data")
plt.plot(x_test,y_test,"o",label="test data")
# hypothesis
plt.plot(x_plot, h(θ, x_plot), color="r", label="your regularized solution")
# formatting
plt.axis([-0.2, 5.2, -1.0, 2.0])
plt.grid(alpha=0.4)
plt.legend()

# Finding λ_min and λ_max
lambda_values = np.linspace(0, 1, 1000)
loss_threshold = 0.08

λ_min = None
λ_max = None

for _λ in lambda_values:
    _θ = optimalθ(_λ)
    l = loss(_θ, x_test, y_test, _λ)
    if l < loss_threshold:
        if λ_min is None:
            λ_min = _λ
        λ_max = _λ

print("λ_min:", λ_min)
print("λ_max:", λ_max)

print("the loss at λ_min is:", loss(optimalθ(λ_min), x_test, y_test, λ_min))
print("the loss at λ_max is:", loss(optimalθ(λ_max), x_test, y_test, λ_max))

### AUTOMATIC TESTS

print('Here we test if the results of `loss` are plausible. What happens for λ=0?')
loss0_train = loss(neθ, x_train, y_train, 0)
loss0_test = loss(neθ, x_test, y_test, 0)

print('REGRESSION WITHOUT REGULARIZATION, i.e. λ=0.')
print('Loss without regularization on training set:', loss0_train)
print('Loss without regularization on test set:', loss0_test)
print('The loss for neθ on the test set should be MUCH larger than on the training set.')
assert loss0_train * 100 < loss0_test
print('Success!')

print('In fact, the loss on the trainig set should be almost zero.')
assert abs(loss0_train) < 1e-5
print('Success!')
print('')

print('The loss should always get larger whenever we increase λ.')
losses = np.array([loss(neθ, x_train, y_train, λ) for λ in 10 ** (np.linspace(-5, 0, 100))])
assert np.all(losses[1:] - losses[:-1] > 0)  # Are the losses getting larger?
print('This seems to be the case!')
print('')

print('REGRESSION WITH REGULARIZATION, we use the λ you selected above.')
lossλ_train = loss(θ, x_train, y_train, λ)
lossλ_test = loss(θ, x_test, y_test, λ)
print('The loss for θ on the test set should be MUCH lower now.')
print('Loss with regularization on training set:', lossλ_train)
print('Loss with regularization on test set:', lossλ_test)
assert lossλ_train * 100 > lossλ_test
print('And it is!\n')
