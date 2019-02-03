"""Staff for training NER models"""

from typing import List, Mapping, Any, Tuple

from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, \
    Bidirectional, Lambda, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.initializers import RandomUniform
import keras.backend as K

from word_casing import get_casing, CASING_PADDING
from char_features import encode_word
from feature_coder import FeatureCoder


def build_model_morph_rnn(num_classes, embed_matrix, word_length, num_chars):
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
        TimeDistributed(Embedding(num_chars + 1,
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
    model.name = 'morph_rnn'
    return model


def make_morpho_inputs(sentences: List[List[str]],
                       word_to_index: Mapping[str, int],
                       characters: List[str],
                       char_cnn_vector_size: int) -> Tuple[List[List[List[Any]]], List[Any]]:
    char_to_index = {ch: i + 1 for i, ch in enumerate(characters)}
    inputs = []
    for sentence in sentences:
        word_indexes = [word_to_index[word.lower()] for word in sentence]
        word_casings = [get_casing(word) for word in sentence]
        char_encodings = [encode_word(word, char_to_index, char_cnn_vector_size) for word in sentence]
        inputs.append([word_indexes, word_casings, char_encodings])
    padding_values = [0, CASING_PADDING, [0] * char_cnn_vector_size]
    return inputs, padding_values


def expand(lst: List[Any]):
    return [[x] for x in lst]


def make_morpho_outputs(train_pos: List[List[str]],
                        train_features: List[List[str]],
                        feature_encoder: FeatureCoder) -> Tuple[List[List[List[int]]], List[Any]]:
    train_morph = feature_encoder.encode(train_pos, train_features)
    outputs = []
    for s in train_morph:
        e = expand(s)
        outputs.append([e, [[0]] + e[:-1], e[1:] + [[0]]])
    return outputs, [[0], [0], [0]]
