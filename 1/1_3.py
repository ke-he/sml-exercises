import matplotlib.pyplot as plt
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

plt.scatter(areas, prices, label="data")
plt.xlabel("Area [m$^2$]")
plt.ylabel("Price [€]")
plt.title("Housing Prices in Amsterdam (2021)")
plt.legend()
plt.grid()

plt.show()
