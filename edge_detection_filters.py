import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random

# Explore the data
mnist_data = pd.read_csv("./datasets/train_mnist.csv")
# print(mnist_data.head())
# print(mnist_data.info())

# 784 features, 1 label. This means that images are 28x28
# print(mnist_data.iloc[0, :])


# Extract a sample, and its features and labels.
def get_sample(idx):
    """
    Extracts a sample from the training data and extracts its features and labels.
    :param idx: int - index of sample.
    :return: sample_features, sample_labels, both flattened pandas arrays.
    """
    try:
        sample_features, sample_labels = mnist_data.iloc[idx, 1:], mnist_data.iloc[idx, 0]
        return sample_features, sample_labels

    except IndexError:
        print("The image is a flattened 28x28 array. The specified index is out of bounds.")


# Transform the sample into a 28x28 numpy array for matplotlib to process.
def transform_image(image):
    arr = []
    # Extracts rows out from the image into the numpy array
    for i in range(0, len(image), 28):
        arr.append(image[i:i+28])

    return np.array(arr)


# Select random sample from the dataset
features, label = get_sample(random.randint(0, 783))
plt.imshow(transform_image(features))
plt.title(label)
plt.show()

