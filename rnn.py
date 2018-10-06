from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.regularizers import L1L2
import numpy as np
import os


def predict(x_test: np.array):

    regressor = create_model(x_test)
    regressor.load_weights(__get_model_path())

    return regressor.predict(x_test)


def train(x_train, y_train):
    regressor = create_model(x_train)
    history = regressor.fit(x_train, y_train, epochs=10, batch_size=x_train.shape[0])
    regressor.save(__get_model_path())
    return history


def create_model(x_data):
    regressor = Sequential()

    regressor.add(SimpleRNN(units=400, return_sequences=True, input_shape=(x_data.shape[1], x_data.shape[2])))
            # , kernel_regularizer=L1L2(0.01), activity_regularizer=L1L2(0.01), recurrent_regularizer=L1L2(0.01)))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units=400, return_sequences=True, activation='relu'))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units=400, return_sequences=True, activation='relu'))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units=200, activation='tanh'))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=x_data.shape[1]))

    # RMSprop optimizer is usually used for rnn
    regressor.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape', 'cosine'])

    return regressor


def __get_model_path():
    return os.path.join(os.path.abspath(os.getcwd()), 'resources', 'gpw_rnn_model.h5')