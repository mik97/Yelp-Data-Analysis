import tensorflow as tf
import tensorflow_hub as hub

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


# NOTA: hub.kerasLayer -> wrappa un SavedModel (scaricato dall'hub) in un keras layer
def build_BERT_model(handle_preprocess, handle_encoder):
    # crea un tensore simbolico rappresentante l'input, necessario per la
    # costruzione iniziale del modello keras
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    # -- creazione preprocessing layer e preprocessing dei dati --
    preprocessing_layer = hub.KerasLayer(
        handle_preprocess, name='preprocessing')
    # frasi processate dal preprocessing che saranno inputs dell'encoder
    encoder_inputs = preprocessing_layer(text_input)

    # -- creazione enconder layer e generazione output --
    #  trainable == true for fine tuning
    encoder = hub.KerasLayer(
        handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)

    # -- def net --
    # dense-> dropout -> output
    # prendiamo in considerazione solo questo output
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    # net = tf.keras.layers.Dense(10, activation="relu", name='dense1')(net)
    net = tf.keras.layers.Dense(1, name='dense2')(net)

    return tf.keras.Model(text_input, net)
