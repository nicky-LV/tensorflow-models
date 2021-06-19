import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from keras.callbacks import EarlyStopping
from collections import Counter


data = pd.read_csv("../datasets/forest_cover_data.csv")

# Exploratory data analysis. Found that data must be standardized due to features having different scales.
# print(data.head())
# print(data.describe())
# print(data.columns)

features = data.iloc[:, 0:len(data)-1]
labels = data['class']

# print(Counter(labels))
# We can see that we have 7 different labels. Having labels in this "LabelEncoded" fashion could introduce bias.
# Why? There is a clear bias 7 > 6 > 5, etc. so there is a bias towards the biggest labels.

labels = LabelBinarizer().fit_transform(labels)

# The percentage of outliers in the dataset is small. We can scale the data as a normal distribution (mean=0, std=1)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2, shuffle=True)

# Use mean/std. of training data on training data only. This prevents leakage between train/test datasets.
x_train = StandardScaler().fit_transform(x_train)

# Use mean/std. of testing data on testing data only.
x_test = StandardScaler().fit_transform(x_test)

model = Sequential()
model.add(InputLayer(input_shape=(55)))
model.add(Dense(36, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(18, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(7, activation='softmax'))

print(model.summary())

early_stopping_obj = EarlyStopping(monitor="accuracy", mode="max", patience=5)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, callbacks=[early_stopping_obj], epochs=5, batch_size=16)
model.save(".")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy {accuracy}")
