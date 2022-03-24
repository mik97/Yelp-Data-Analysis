import keras_tuner as kt
import constants as const

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential


def get_rnn_builder(drop, units, lrate, optimizer, embedding_layer):
    ''' Returns a function for build rnn model with custon hyperparams values'''

    def rnn_builder(hp):
        # Define the hyperparams
        dropout = hp.Choice("dropout", drop)
        lstm_units = hp.Choice("units", units)
        lr = hp.Choice("lr", lrate)

        model = Sequential()

        model.add(embedding_layer)  # the embedding layer
        model.add(LSTM(lstm_units, dropout=dropout))
        model.add(Dense(1, activation='sigmoid'))

        opt = optimizer(learning_rate=lr)

        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    return rnn_builder
