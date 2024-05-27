# Very popular packages in the Python ecosystem.
import numpy as np  # For fast computations on numerical data
import matplotlib.pyplot as plt  # For visualizations of said data
import pandas as pd  # For saving/loading data as dataframes (convenience)

# Other
import sklearn
import sklearn.linear_model
from sklearn import preprocessing

"""
We load the SignMNIST dataset, which contains hand gestures of letters from the American Sign Language. 
Normally this data set contains several thousand images, but we only take the first 24 unique hand signs.
"""
with np.load("sign_mnist.npz") as f:
    images, labels = f["x_train"], f["y_train"]

# For this exercise we only need 24 different signs. We prepare them for you!
_, indices = np.unique(labels, return_index=True)  # Use the np.unique to find the position of the unique labels
images = images[indices]  # We select the 24 unique digits

"""
Part A: Plotting a single image.
You don't have to do this part, it is just here to help you for the actual exercise below.
"""
# Plot the first image in the dataset
image = images[0]  # Select the first image
print("Image shape:", image.shape)  # A 28x28 pixel image

# Use the matplotlib function imshow to plot it.a
plt.imshow(image, cmap="gray")
plt.show()
