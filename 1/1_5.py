import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
We load the amsterdam.csv data set with Pandas and extract the residental price and residental area as a NumPy array.
"""
# Source: https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction
df = pd.read_csv("amsterdam.csv")  # Load CSV with pandas
# We exclude some outliers for better visualization
df = df.loc[(df["Area"] < 350) & (df["Price"] < 3e6)]
areas = df["Area"].to_numpy()  # Save area as a numpy array
prices = df["Price"].to_numpy()  # Save price as a numpy array

def loss(theta):
    m = len(areas)
    predictions = theta[0] + theta[1] * areas
    return (1 / (2 * m)) * np.sum((predictions - prices) ** 2)


def gradient(θ):  # DO NOT CHANGE
    ϵ = 1e-4
    ϵ0 = np.array([ϵ,0])
    ϵ1 = np.array([0,ϵ])
    dθ0 = (loss(θ + ϵ0) - loss(θ - ϵ0))/(2*ϵ)  # Derivative with respect to θ[0]
    dθ1 = (loss(θ + ϵ1) - loss(θ - ϵ1))/(2*ϵ)  # Derivative with respect to θ[1]
    return np.array([dθ0, dθ1])


alpha = 1e-4
num_iters = 1000

thetaGD = np.zeros(2)


def gradient_descent(theta):
    for _ in range(num_iters):
        theta = theta - alpha * gradient(theta)
    return theta


thetaGD = gradient_descent(thetaGD)


def h2(x):
    return thetaGD[0] + thetaGD[1] * x


def main():
    # Plot the data
    plt.scatter(areas, prices, label="data")
    plt.xlabel("Area [m$^2$]")
    plt.ylabel("Price [€]")
    plt.title("Housing Prices in Amsterdam (2021)")
    plt.legend()
    plt.grid()

    # Plot the linear regressionÍ
    x_vals = np.linspace(0, 350, 100)
    y_vals = h2(x_vals)
    plt.plot(x_vals, y_vals, label="gradient descent", color="orange")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
