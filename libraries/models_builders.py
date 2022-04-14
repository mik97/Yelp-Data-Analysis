import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential


def get_rnn_builder(drop, units, lrate, optimizer, embedding_layer, output_shape, loss, activation=None, metrics=[]):
    ''' Returns a function for build rnn model with custon hyperparams values'''

    def rnn_builder(hp):
        dropout = hp.Choice("dropout", drop)
        lstm_units = hp.Choice("units", units)
        lr = hp.Choice("lr", lrate)

        model = Sequential()

        model.add(embedding_layer)  # the embedding layer
        model.add(LSTM(lstm_units, dropout=dropout))
        model.add(Dense(output_shape, activation=activation))

        opt = optimizer(learning_rate=lr)
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=metrics)

        return model

    return rnn_builder


def build_BERT_model(handle_preprocess, handle_encoder, output_shape, activation=None):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    preprocessing_layer = hub.KerasLayer(
        handle_preprocess, name='preprocessing')

    encoder_inputs = preprocessing_layer(text_input)

    encoder = hub.KerasLayer(
        handle_encoder, trainable=True, name='BERT_encoder')

    outputs = encoder(encoder_inputs)

    net = outputs['pooled_output']

    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(
        output_shape, activation=activation, name='dense1')(net)

    return tf.keras.Model(text_input, net)
