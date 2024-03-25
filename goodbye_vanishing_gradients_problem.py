from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.initializers import RandomNormal, glorot_normal, he_normal
from tensorflow.random import normal
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam 
from keras.utils import plot_model
import pandas as pd

class NoVanishingGradientsInit(RandomNormal):
    def __init__(self, mean = None, stddev = 0.1):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        if len(shape) == 2:
            fan_in = shape[0]
        else:
            receptive_field_size = 1
            for dim in shape[:-2]:
                receptive_field_size *= dim
            fan_in = shape[-2] * receptive_field_size

        return normal(shape, mean=-8.0 / fan_in, stddev=self.stddev, dtype=dtype)

def get_train_and_test_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(50000,32,32,3)/255
    X_test = X_test.reshape(10000,32,32,3)/255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test

def define_model(initializer):
    model = Sequential()
    model.add(Input(shape=(32,32,3)))
    for conv_layer_size in [64, 32, 16]:
        model.add(Conv2D(conv_layer_size, kernel_size=3, kernel_initializer=initializer, padding='same', activation="sigmoid"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
    model.add(Flatten())
    for extra_layers in range(10):
        model.add(Dense(100, kernel_initializer=initializer, activation="sigmoid"))
        model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    return model

X_train, y_train, X_test, y_test = get_train_and_test_data()

for initializer in [NoVanishingGradientsInit(), he_normal(), glorot_normal()]:
    print(initializer)
    model = define_model(initializer)
    model.compile(optimizer = SGD(learning_rate = 0.5), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
