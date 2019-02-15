"""Staff for training relation extraction models"""

from typing import List, Mapping, Any, Tuple

from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, \
    Bidirectional, Lambda, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.initializers import RandomUniform, Identity
from padding import pad_sequence_left
import keras.backend as K

from word_casing import get_casing, CASING_PADDING
from char_features import encode_word
from feature_coder import FeatureCoder


def make_rel_ext_inputs(words: List[List[str]],
                        word_to_index: Mapping[str, int],
                        pos_tags: List[List[str]],
                        pos_to_index: Mapping[str, int],
                        dep_labels: List[List[str]],
                        dep_to_index: Mapping[str. int],
                        branch_indices_1: List[List[int]],
                        branch_indices_2: List[List[int]]) -> Tuple[List[List[List[Any]]], List[Any]]:
    inputs = []
    for sent_words, sent_pos_tags, sent_deps, br1, br2 in \
            zip(words, pos_tags, dep_labels, branch_indices_1, branch_indices_2):
        word_indices = [word_to_index[word] for word in sent_words]
        pos_tag_indices = [pos_to_index[pos_tag] for pos_tag in sent_pos_tags]
        dep_indices = [dep_to_index[dep] for dep in sent_deps]
        sent_len = len(word_indices)
        mask1 = pad_sequence_left([1.0] * len(br1), sent_len, 0)
        mask2 = pad_sequence_left([1.0] * len(br2), sent_len, 0)
        br1_pad = pad_sequence_left(br1, sent_len, 0.)
        br2_pad = pad_sequence_left(br2, sent_len, 0.)

        inputs.append([word_indices, pos_tag_indices, dep_indices, br1_pad, br2_pad, mask1, mask2])
    padding_values = [0, 0, 0, 0, 0, 0., 0.]
    return inputs, padding_values


def build_rel_ext_model(num_classes, embed_matrix, pos_classes_num):
    words_indices_input = Input(shape=(None,), name='words_indices_input')
    pos_tag_indices_input = Input(shape=(None,), name='pos_tag_indices_input')
    dep_tag_indices_input = Input(shape=(None,), name='dep_tag_indices_input')
    branch_1_indices = Input(shape=(None,), name='branch_1_indices')
    branch_2_indices = Input(shape=(None,), name='branch_2_indices')
    branch_1_mask = Input(shape=(None,), name='branch_1_mask')
    branch_2_mask = Input(shape=(None,), name='branch_2_mask')
    word_embeddings_out = Embedding(embed_matrix.shape[0],
                                    embed_matrix.shape[1],
                                    weights=[embed_matrix],
                                    trainable=False,
                                    name='word_embeddings')(words_indices_input)
    pos_embeddings_out = Embedding(pos_classes_num,
                                   pos_classes_num,
                                   trainable=True,
                                   embeddings_initializer=Identity())(pos_tag_indices_input)

    concatenated_out = concatenate([word_embeddings_out, pos_embeddings_out], name='concat')
    bilstm_out = Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.25, return_sequences=True), name='bilstm')(
        concatenated_out)
    td_fc_out = TimeDistributed(Dense(num_classes + 1, activation='softmax'), name='tdfc')(bilstm_out)

    gather_layer = Lambda(lambda t: K.gather(t[0], t[1]))\

    branch_1_input = gather_layer([td_fc_out, branch_1_indices])
    branch_2_input = gather_layer([td_fc_out, branch_2_indices])


    Bidirectional(LSTM)

    lstm_model = Model([words_indices_input, pos_tag_indices_input], td_fc_out)
    lstm_model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy')
    lstm_model.name = 'bilstm'

    return lstm_model
