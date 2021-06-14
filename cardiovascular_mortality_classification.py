import pandas as pd
import numpy as np
import itertools

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

data = pd.read_csv("./datasets/heart_failure_clinical_records_dataset.csv")

# Explore data distribution of cardiovascular mortality.
# print(Counter(data["DEATH_EVENT"]))
# 1 is a mortality event. 0 is not.

features = data.iloc[:, 0:len(data.columns)-1]
labels = data["DEATH_EVENT"]

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)

# Scale numerical data.
numerical_cols = features.select_dtypes(include=["float64", "int64"])
ct = ColumnTransformer([("Scale numerical data", StandardScaler(), numerical_cols.columns)])

# Fit and transform training data.
x_train = ct.fit_transform(x_train)

# Transform test data only.
x_test = ct.transform(x_test)

# One-hot-encode the labels, for cross-entropy loss calculations further on.
# Since we one-hot-encoded, we must use categorical_crossentropy for our loss fn.
y_train, y_test = to_categorical(y_train, dtype="int64"), to_categorical(y_test, dtype="int64")

# Create sequential model
model = Sequential()
model.add(InputLayer(input_shape=(features.shape[1],)))
model.add(Dense(12, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model to use categorical_crossentropy, accuracy as metric with the Adam optimizer (default learning rate)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Fit model to training data
model.fit(x_train, y_train, epochs=40, batch_size=16, verbose=0)

# Evaluate the loss / accuracy. 80% accurate.
loss, accuracy = model.evaluate(x=x_test, y=y_test)
# print(loss, accuracy)

# Predict labels
y_estimated = np.argmax(model.predict(x=x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report. Get f1-score which represents false negatives.
print(classification_report(y_estimated, y_true))
