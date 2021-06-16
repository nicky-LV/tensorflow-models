import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping

# Load training / testing data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshaping data to fit input layer.
input_shape = (28, 28, 1)
x_train, x_test = np.reshape(x_train, (len(x_train), 28, 28, 1)), np.reshape(x_test, (len(x_test), 28, 28, 1))


# Fn for creating the model
def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=12, kernel_size=3, strides=2, activation='relu'))
    model.add(Conv2D(filters=12, kernel_size=3, strides=2, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    return model


# Early stopping to save on computation time.
early_stopper = EarlyStopping(monitor='loss', mode='min', patience=5)
model = create_model()
# Compile model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fit model to training data.
model.fit(x=x_train, y=y_train, verbose=1, epochs=40, callbacks=[early_stopper], batch_size=10)

loss, accuracy = model.evaluate(x=x_test, y=y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
