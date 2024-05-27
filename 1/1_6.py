import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model

"""
We load the amsterdam.csv data set with Pandas and extract the residential price and residential area as a NumPy array.
"""
# Source: https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction
df = pd.read_csv("amsterdam.csv")  # Load CSV with pandas
# We exclude some outliers for better visualization
df = df.loc[(df["Area"] < 350) & (df["Price"] < 3e6)]
areas = df["Area"].to_numpy().reshape(-1, 1)  # Save area as a numpy array and reshape to 2D
prices = df["Price"].to_numpy()  # Save price as a numpy array

# Define and train a LinearRegression model from scikit-learn here using areas and prices.
model = sklearn.linear_model.LinearRegression()
model.fit(areas, prices)

# Save the model parameters
θ0 = model.intercept_
θ1 = model.coef_[0]


def h3(x):
    # This converts x to an array, if it is a float or integer.
    if isinstance(x, (int, float)):
        x = np.array([x])
    x = np.array(x).reshape(-1, 1)

    # Use the fitted model here.
    result = model.predict(x)
    return result


# Test the function
testarray = np.random.rand(10) * 1000
assert type(h3(testarray)) is np.ndarray
assert len(h3(testarray)) == 10
assert np.isscalar(θ0)
assert np.isscalar(θ1)


# Plot the data and the regression line
def main():
    # Plot the data
    plt.scatter(areas, prices, label="Data")
    plt.xlabel("Area [m$^2$]")
    plt.ylabel("Price [€]")
    plt.title("Housing Prices in Amsterdam (2021)")
    plt.legend()
    plt.grid()

    # Plot the linear regression
    x_values = np.linspace(0, 350, 100)
    y_values = h3(x_values)
    plt.plot(x_values, y_values, color='orange', label="Linear Regression")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
