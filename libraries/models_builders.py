import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential


def rnn_builder(hp, dropout, units, lr, embedding_layer):
    # Define the hyperparams
    dropout = hp.Choice("dropout", [0.2, 0.5])
    lstm_units = hp.Choice("units", [15, 20, 50])
    lr = hp.Choice("lr", [0.01, 0.001])

    model = Sequential()
    model.add(embedding_layer)  # the embedding layer
    model.add(LSTM(lstm_units, dropout=dropout))
    # if dropout:
    #     model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
