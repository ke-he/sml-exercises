# Very popular packages in the Python ecosystem.
import numpy as np  # For fast computations on numerical data
import matplotlib.pyplot as plt  # For visualizations of said data
import pandas as pd  # For saving/loading data as dataframes (convenience)

# Other
import sklearn
import sklearn.linear_model
from sklearn import preprocessing

"""
We load the amsterdam.csv data set with Pandas and extract the residental price and residental area as a NumPy array.
"""
# Source: https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction
df = pd.read_csv("amsterdam.csv")  # Load CSV with pandas
# We exclude some outliers for better visualization
df = df.loc[(df["Area"] < 350) & (df["Price"] < 3e6)]
areas = df["Area"].to_numpy()  # Save area as a numpy array
prices = df["Price"].to_numpy()  # Save price as a numpy array

X = np.column_stack((np.ones(areas.shape[0]), areas))
y = prices

theta = np.linalg.solve(X.T @ X, X.T @ y)


def h1(x):
    return theta[0] + theta[1] * x


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
    y_vals = h1(x_vals)
    plt.plot(x_vals, y_vals, label="normal equation", color="orange")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
