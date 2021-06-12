import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, CSVLogger

"""
Predict the chance of admission to a university regressed from historical data.

This is a regression task because our label "Chance of Admit" is a continuous variable between 0 and 1.
"""

# Todo: Use grid-search / random-search to tune hyper-parameters to improve accuracy metric.

# Load & explore data dataset
dataset = pd.read_csv("./datasets/admissions_data.csv")
# print(dataset.head())
# print(dataset.columns)

# Serial No. will not be used as a feature in our classification, so it's removed.
dataset.drop(axis=1, columns="Serial No.", inplace=True)

# Split dataset into labels (chance of admittance) and features (all other columns)
features = dataset.iloc[:, 0: len(dataset.columns)-1]
labels = dataset.iloc[:, -1]

# All of our columns in the dataset are numerical, so we do not need One-Hot-Encoding to transform these
# categorical variables into numerical ones.
# non_numerical_cols = dataset.select_dtypes(exclude=["float64", "int64"]).columns
# print(non_numerical_cols)

# Split data into training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=4)

# We must standardize our data as some of our features have different scales.
# This introduces an implicit bias / weight of certain features with a large scale - which may not be necessarily true.
numerical_cols = features.select_dtypes(include=["float64", "int64"])
ct = ColumnTransformer([("Standard Scalar", StandardScaler(), numerical_cols.columns)])

# Scale training features
scaled_features_train = ct.fit_transform(train_features)
# Scale testing features
scaled_features_test = ct.fit_transform(test_features)

# Model definition
model = Sequential()
# Input layer with n nodes (where n = num of features)
model.add(InputLayer(input_shape=(features.shape[1],)))
# Hidden layer(s)
model.add(Dense(48, activation="relu"))
model.add(Dense(48, activation="relu"))
# Output layer with a single node, due to regression

model.add(Dense(1))

# Early stopping to improve performance and avoid divergence.
# We monitor the loss value and mode = "min" as we want to minimize our loss.
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3, verbose=1)
# Compile and fit model to training data.
model.compile(loss="mse", metrics="mae", optimizer=Adam(learning_rate=0.01))
csv_logger = CSVLogger('./logging/acceptance_admissions_logging.csv')
model.fit(scaled_features_train, train_labels, epochs=15, verbose=1, callbacks=[early_stopping, csv_logger])

# Score the model
model_prediction = model.predict(scaled_features_test)
score = r2_score(test_labels, model_prediction)
print(f"Score of model: {score}")

# Evaluate errors
mean_squared_error, mean_absolute_error = model.evaluate(scaled_features_test, test_labels)

# Plot errors against epochs for hyper-parameter tuning.
logs = pd.read_csv("./logging/acceptance_admissions_logging.csv")
x_values = logs['epoch']
y_values = logs['loss']

plt.xlabel("Epochs")
plt.ylabel("Loss")
fig = plt.plot(x_values, y_values)
plt.savefig("./logging/loss_graph.png")