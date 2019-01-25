"""Строим модель"""

from typing import List, Dict

from keras.layers.merge import concatenate
from keras.optimizers import nadam
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, \
    Bidirectional, Lambda, BatchNormalization
from keras.layers.embeddings import Embedding
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.initializers import RandomUniform
from data_generator import DataGenerator
import keras.backend as K


def build_model_char_cnn_lstm_crf(n_classes, embed_matrix, word_length, n_chars):
    words_indices_input = Input(shape=(None,), name='words_indices_input')
    word_embeddings_out = Embedding(embed_matrix.shape[0],
                                    embed_matrix.shape[1],
                                    weights=[embed_matrix],
                                    trainable=False,
                                    name='word_embeddings')(words_indices_input)

    casings_input = Input(shape=(None, 6),
                          name='casings_input')

    chars_input = Input(shape=(None, word_length),
                        name='chars_input')
    char_embeddings_out = \
        TimeDistributed(Embedding(n_chars,
                                  32,
                                  embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
                        name='char_embeddings')(chars_input)

    char_dropout_out = Dropout(0.5, name='char_dropout')(char_embeddings_out)

    char_conv1d_out = TimeDistributed(Conv1D(kernel_size=3,
                                             filters=32,
                                             padding='same',
                                             activation='tanh',
                                             strides=1,
                                             name='char_conv1d'))(char_dropout_out)

    char_maxpool_out = TimeDistributed(MaxPooling1D(word_length), name='char_maxpool')(char_conv1d_out)
    char_flat_out = TimeDistributed(Flatten(), name='char_flat')(char_maxpool_out)
    char_flat_dropout_out = Dropout(0.5, name='char_flat_dropout')(char_flat_out)

    concatenated_out = concatenate([word_embeddings_out, casings_input, char_flat_dropout_out], name='concat')
    bilstm_out = Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.25, return_sequences=True), name='bilstm')(
        concatenated_out)
    td_fc_out = TimeDistributed(Dense(50,), name='tdfc')(bilstm_out)
    relu_out = Activation('relu', name='activation')(td_fc_out)
    crf_out = CRF(n_classes, name='crf')(relu_out)
    cnn_lstm_crf_model = Model([words_indices_input, casings_input, chars_input], crf_out)
    cnn_lstm_crf_model.compile(optimizer='nadam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    return cnn_lstm_crf_model


def build_model_char_cnn_lstm(num_classes, embed_matrix, word_length, num_chars):
    words_indices_input = Input(shape=(None,), name='words_indices_input')
    word_embeddings_out = Embedding(embed_matrix.shape[0],
                                    embed_matrix.shape[1],
                                    weights=[embed_matrix],
                                    trainable=False,
                                    name='word_embeddings')(words_indices_input)

    casings_input = Input(shape=(None, 6),
                          name='casings_input')

    chars_input = Input(shape=(None, word_length),
                        name='chars_input')
    char_embeddings_out = \
        TimeDistributed(Embedding(num_chars,
                                  32,
                                  embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
                        name='char_embeddings')(chars_input)

    char_dropout_out = Dropout(0.5, name='char_dropout')(char_embeddings_out)

    char_conv1d_out = TimeDistributed(Conv1D(kernel_size=3,
                                             filters=32,
                                             padding='same',
                                             activation='tanh',
                                             strides=1,
                                             name='char_conv1d'))(char_dropout_out)

    char_maxpool_out = TimeDistributed(MaxPooling1D(word_length), name='char_maxpool')(char_conv1d_out)
    char_flat_out = TimeDistributed(Flatten(), name='char_flat')(char_maxpool_out)
    char_flat_dropout_out = Dropout(0.5, name='char_flat_dropout')(char_flat_out)

    concatenated_out = concatenate([word_embeddings_out, casings_input, char_flat_dropout_out], name='concat')
    bilstm_out = Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.25, return_sequences=True), name='bilstm')(
        concatenated_out)
    td_fc_out = TimeDistributed(Dense(num_classes, activation='softmax'), name='tdfc')(bilstm_out)
    cnn_lstm_model = Model([words_indices_input, casings_input, chars_input], td_fc_out)
    cnn_lstm_model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy')
    cnn_lstm_model.name = 'char_cnn_bilstmm'
    return cnn_lstm_model


def build_model_predict_neighbour(num_classes, embed_matrix, word_length, num_chars):
    words_indices_input = Input(shape=(None,), name='words_indices_input')
    word_embeddings_out = Embedding(embed_matrix.shape[0],
                                    embed_matrix.shape[1],
                                    weights=[embed_matrix],
                                    trainable=False,
                                    name='word_embeddings')(words_indices_input)

    casings_input = Input(shape=(None, 6),
                          name='casings_input')

    chars_input = Input(shape=(None, word_length),
                        name='chars_input')
    char_embeddings_out = \
        TimeDistributed(Embedding(num_chars,
                                  32,
                                  embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
                        name='char_embeddings')(chars_input)

    char_dropout_out = Dropout(0.5, name='char_dropout')(char_embeddings_out)

    char_conv1d_out = TimeDistributed(Conv1D(kernel_size=3,
                                             filters=32,
                                             padding='same',
                                             activation='tanh',
                                             strides=1,
                                             name='char_conv1d'))(char_dropout_out)

    char_maxpool_out = TimeDistributed(MaxPooling1D(word_length), name='char_maxpool')(char_conv1d_out)
    char_flat_out = TimeDistributed(Flatten(), name='char_flat')(char_maxpool_out)
    char_flat_dropout_out = Dropout(0.5, name='char_flat_dropout')(char_flat_out)

    concatenated_out = concatenate([word_embeddings_out, casings_input, char_flat_dropout_out], name='concat')
    forward_lstm_out = LSTM(128, dropout=0.5, recurrent_dropout=0.25, return_sequences=True)(concatenated_out)
    backward_lstm_out = LSTM(128, dropout=0.5, recurrent_dropout=0.25, return_sequences=True, go_backwards=True)(
        concatenated_out)
    reverse = Lambda(lambda x: K.reverse(x, axes=1))
    bilstm_out = concatenate([forward_lstm_out, reverse(backward_lstm_out)])
    # Предсказываем текущую метку по выходам bilstm
    bilstm_2_out = Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.25, return_sequences=True))(bilstm_out)
    fc_out = TimeDistributed(Dense(128, activation='relu'), name='fc_1')(bilstm_2_out)
    bn_out = TimeDistributed(BatchNormalization(axis=1))(fc_out)
    fc_2_out = TimeDistributed(Dense(num_classes, activation='softmax'), name='fc_2')(bn_out)
    # forward lstm предсказывает следующую метку справа
    fc_fw_out = TimeDistributed(Dense(128, activation='relu'), name='fc_fw_1')(forward_lstm_out)
    fc_fw_2_out = TimeDistributed(Dense(num_classes, activation='softmax', name='fc_fw_2'))(fc_fw_out)
    # backward lstm предсказывает следующую метку слева
    fc_bw_out = TimeDistributed(Dense(128, activation='relu'), name='fc_bw_1')(backward_lstm_out)
    fc_bw_2_out = TimeDistributed(Dense(num_classes, activation='softmax', name='fc_bw_2'))(fc_bw_out)

    model = Model([words_indices_input, casings_input, chars_input], [fc_2_out, fc_bw_2_out, fc_fw_2_out])
    model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy')

    return model


def predict(model: Model,
            data_generator: DataGenerator,
            idx_to_label: Dict[int, str] = None):
    prediction = [None] * sum(len(data_generator.get_indices_and_lengths(i)) for i in range(len(data_generator)))
    for i in range(len(data_generator)):
        model_input = data_generator[i]
        indices_and_lengths = data_generator.get_indices_and_lengths(i)
        softmax_prediction = model.predict(model_input)
        if softmax_prediction.shape == 4:
            softmax_prediction = softmax_prediction[0]
        class_prediction = softmax_prediction.argmax(axis=-1)
        for sent_pred, (idx, length) in zip(class_prediction, indices_and_lengths):
            sent_labels = [idx_to_label[l] for l in sent_pred[-length:]] if idx_to_label else sent_pred[-length:]
            prediction[idx] = sent_labels
    # assert all(prediction)
    return prediction

