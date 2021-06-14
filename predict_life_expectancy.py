import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

"""
Predict the life-expectancy from countries given a lot of other socio-economic & related factors.
"""
# Read dataset
dataset = pd.read_csv("life_expectancy.csv")

# Remove the "Country" column - our predictions should be generalized and should not be biased to a specific country.

dataset.drop(columns=["Country"], axis=1, inplace=True)

# Get our labels (what we want to classify). This is the life-expectancy column.

labels = dataset.iloc[:, -1]

# Get our features (factors that we use to classify). This is all of our features.

features = dataset.iloc[:, 0: len(dataset.columns) - 1]

# One-Hot-Encoding of categorical columns. E.g. "Status" column must not be "Developing" but a numerical feature (as we can't process non-numerical data).

one_hot_encoded_features = pd.get_dummies(features)

# Train test split, 20% = testing data. Random seed = 4.
features_train, features_test, labels_train, labels_test = train_test_split(one_hot_encoded_features, labels, test_size=0.2, random_state = 4)

# Standardize numerical features < read more about this.

ct = ColumnTransformer([("StandardScaling", StandardScaler(), one_hot_encoded_features.columns)], remainder='passthrough')

# Standard Scale train features
features_train_scaled = ct.fit_transform(features_train)

# Standard Scale test features
features_test_scaled = ct.fit_transform(features_test)

# Instantiate sequential model
my_model = Sequential()

# Add input layer to model
input_layer = InputLayer(input_shape=(one_hot_encoded_features.shape[1], ))
my_model.add(input_layer)

# Add hidden layer to model
my_model.add(Dense(24, activation="relu"))

# Add output layer to model
my_model.add(Dense(1))

# Summary of model
print(my_model.summary())
