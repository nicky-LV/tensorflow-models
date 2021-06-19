import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy

# Normalize pixel values within images
image_augmentation = ImageDataGenerator(rescale=1/255)

# Training data iterator
training_iterator = image_augmentation.flow_from_directory(directory="../datasets/Covid19-dataset/train",
                                                       target_size=(256, 256),
                                                       class_mode="sparse",
                                                       color_mode="grayscale",
                                                       batch_size=10)

# print(training_data.image_shape)

# Testing data iterator
testing_iterator = image_augmentation.flow_from_directory(directory="../datasets/Covid19-dataset/test",
                                                      target_size=(256, 256),
                                                      class_mode="sparse",
                                                      color_mode="grayscale",
                                                      batch_size=10)

# Early stopping to reduce computation and over-fitting.
early_stopping_callback = EarlyStopping(monitor="accuracy", mode="max", patience=10)

# Model definition
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(filters=12, kernel_size=3, strides=2, activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Conv2D(filters=8, kernel_size=2, strides=2, activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=3))
model.add(Flatten())
model.add(Dense(12, activation="relu"))
model.add(Dense(3, activation="softmax"))

# Model summary / compilation.
print(model.summary())
model.compile(optimizer="adam", metrics=["accuracy"], loss=SparseCategoricalCrossentropy())

# Fitting model to training data and retrieving history object for data retrieval & plotting.
history = model.fit(training_iterator, validation_data=testing_iterator, epochs=40, callbacks=[early_stopping_callback], verbose=1, batch_size=15)
loss, accuracy = model.evaluate(testing_iterator)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Epochs vs accuracy plot
plt.plot(x=history.history['epochs'], y=history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()