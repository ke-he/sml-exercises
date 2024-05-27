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
Part B: Plotting all hand signs.
This is the actual exercise.
"""

# Plot all 24 (twentyfour) 28x28 `images` with matplotlib. (Iterate over all images and use imshow)
# You might have to iterate over all axes prepared below. (Google if you don't understand what that means)

fig, axes = plt.subplots(6, 4)  # This might help. (Prepares a 6x4 empty plot)

for image, ax in zip(images, axes.flat):
    ax.imshow(image, cmap="gray")
    ax.axis("off")

plt.show()
