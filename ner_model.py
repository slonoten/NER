"""Staff for training NER models"""

from typing import List, Mapping, Any, Dict, Tuple

import numpy as np

from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, \
    Bidirectional
from keras.layers.embeddings import Embedding
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.initializers import RandomUniform

from word_casing import get_casing, CASING_PADDING
from char_features import encode_word


def build_char_cnn_lstm_graph(num_classes, embed_matrix, word_length, num_chars):
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
        TimeDistributed(Embedding(num_chars + 1,  # plus one for padding
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

    return [words_indices_input, casings_input, chars_input], bilstm_out


def build_model_char_cnn_lstm(num_classes, embed_matrix, word_length, num_chars):
    inputs, bilstm_out = build_char_cnn_lstm_graph(num_classes, embed_matrix, word_length, num_chars)
    td_fc_out = TimeDistributed(Dense(num_classes + 1, activation='softmax'), name='tdfc')(bilstm_out)
    cnn_lstm_model = Model(inputs, td_fc_out)
    cnn_lstm_model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy')
    cnn_lstm_model.name = 'char_cnn_bilstm'
    return cnn_lstm_model


def build_model_char_cnn_lstm_crf(num_classes, embed_matrix, word_length, num_chars):
    inputs, bilstm_out = build_char_cnn_lstm_graph(num_classes, embed_matrix, word_length, num_chars)
    td_fc_out = TimeDistributed(Dense(50,), name='tdfc')(bilstm_out)
    relu_out = Activation('relu', name='activation')(td_fc_out)
    crf_out = CRF(num_classes + 1, name='crf')(relu_out)
    cnn_lstm_crf_model = Model(inputs, crf_out)
    cnn_lstm_crf_model.compile(optimizer='nadam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    cnn_lstm_crf_model.name = 'char_cnn_bilstm_crf'
    return cnn_lstm_crf_model


def make_ner_inputs(sentences: List[List[str]],
                    word_to_index: Mapping[str, int],
                    characters: List[str],
                    char_cnn_vector_size: int) -> Tuple[List[List[List[Any]]], List[Any]]:
    char_to_index = {ch: i + 1 for i, ch in enumerate(characters)}
    inputs = []
    for sentence in sentences:
        word_indexes = [word_to_index[word] for word in sentence]
        word_casings = [get_casing(word) for word in sentence]
        char_encodings = [encode_word(word, char_to_index, char_cnn_vector_size) for word in sentence]
        inputs.append([word_indexes, word_casings, char_encodings])
    padding_values = [0, CASING_PADDING, [0] * char_cnn_vector_size]
    return inputs, padding_values


def to_one_hot(indices: Any, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[indices]


def make_ner_outputs(sentences_labels: List[List[str]],
                     label_to_index: Dict[str, int]) -> Tuple[List[List[List[int]]], List[Any]]:
    return [[[[label_to_index[label]] for label in sentence]] for sentence in sentences_labels], [0]


def make_ner_one_hot_outputs(sentences_labels: List[List[str]],
                             label_to_index: Dict[str, int]) -> Tuple[List[List[List[int]]], List[Any]]:
    num_classes = len(label_to_index)
    one_hot_matrix = [[0] * i + [1] + [0] * (num_classes - i - 1) for i in range(num_classes)]
    return [[[one_hot_matrix[label_to_index[label]] for label in sentence]] for sentence in sentences_labels], \
           [one_hot_matrix[0]]

