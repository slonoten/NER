"""Staff for training relation extraction models"""

from typing import List, Mapping, Any, Tuple

from keras.layers.merge import concatenate, multiply
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, \
    Bidirectional, Lambda, BatchNormalization, RepeatVector
from keras.layers.embeddings import Embedding
from keras.initializers import RandomUniform, Identity
from padding import pad_sequence_left
import keras.backend as K
import tensorflow as tf

from word_casing import get_casing, CASING_PADDING
from char_features import encode_word
from feature_coder import FeatureCoder


def make_rel_ext_inputs(words: List[List[str]],
                        word_to_index: Mapping[str, int],
                        pos_tags: List[List[str]],
                        pos_to_index: Mapping[str, int],
                        ent_tags: List[List[str]],
                        ent_to_index: Mapping[str, int],
                        dep_labels: List[List[str]],
                        dep_to_index: Mapping[str, int],
                        branch_indices_1: List[List[int]],
                        branch_indices_2: List[List[int]]) -> Tuple[List[List[List[Any]]], List[Any]]:
    inputs = []
    for sent_words, sent_pos_tags, sent_ent_tags, sent_deps, br1, br2 in \
            zip(words, pos_tags, ent_tags, dep_labels, branch_indices_1, branch_indices_2):
        word_indices = [word_to_index[word] for word in sent_words]
        pos_tag_indices = [pos_to_index[pos_tag] for pos_tag in sent_pos_tags]
        ent_tag_indices = [ent_to_index[ent_tag] for ent_tag in sent_ent_tags]
        dep_indices = [dep_to_index[dep] for dep in sent_deps]
        sent_len = len(word_indices)
        br1_pad = pad_sequence_left(br1, sent_len, -1)
        br2_pad = pad_sequence_left(br2, sent_len, -1)

        inputs.append([word_indices, pos_tag_indices, ent_tag_indices, dep_indices, br1_pad, br2_pad])  # dep_indices,
    padding_values = [0, 0, 0, 0, 0, 0]
    return inputs, padding_values


def gather(layer_input: List[Any]) -> Any:
    value_tensor, indices_tensor = layer_input

    pad_dims = tf.concat([tf.reshape(tf.shape(value_tensor)[0], shape=(1,)),
                          tf.constant([1]),
                          tf.reshape(tf.shape(value_tensor)[2], shape=(1,))], axis=0)

    pad_tensor = tf.fill(pad_dims, .0)

    value_and_pad_tensor = tf.concat([value_tensor, pad_tensor], axis=1)

    # create the row index with tf.range
    row_idx = tf.reshape(tf.range(tf.shape(indices_tensor)[0]), (-1, 1))

    row_indices_dims = tf.concat([tf.constant([1]),
                                  tf.reshape(tf.shape(indices_tensor)[1], shape=(1,))], axis=0)

    row_indices = tf.tile(row_idx, row_indices_dims)
    # stack with column indices
    idx = tf.stack([row_indices, indices_tensor], axis=-1)
    # extract the elements with gather_nd
    return tf.gather_nd(value_and_pad_tensor, idx)


def output_shape_of_gather(input_shape):
    return input_shape[0]


def build_rel_ext_model(num_classes, embed_matrix, pos_classes_num, ent_class_num, dep_classes_num):
    words_indices_input = Input(shape=(None,), name='words_indices_input')
    pos_tag_indices_input = Input(shape=(None,), name='pos_tag_indices_input')
    ent_tag_indices_input = Input(shape=(None,), name='ent_tag_indices_input')
    dep_tag_indices_input = Input(shape=(None,), name='dep_tag_indices_input')
    branch_1_indices = Input(shape=(None,), name='branch_1_indices', dtype='int32')
    branch_2_indices = Input(shape=(None,), name='branch_2_indices', dtype='int32')
    word_embeddings_out = Embedding(embed_matrix.shape[0],
                                    embed_matrix.shape[1],
                                    weights=[embed_matrix],
                                    trainable=False,
                                    name='word_embeddings')(words_indices_input)
    pos_embeddings_out = Embedding(pos_classes_num,
                                   pos_classes_num,
                                   trainable=True,
                                   embeddings_initializer=Identity())(pos_tag_indices_input)

    ent_embeddings_out = Embedding(ent_class_num, ent_class_num, trainable=True,
                                   embeddings_initializer=Identity())(ent_tag_indices_input)

    dep_embeddings_out = Embedding(dep_classes_num, dep_classes_num, trainable=True,
                                   embeddings_initializer=Identity())(dep_tag_indices_input)

    concatenated_out = concatenate([word_embeddings_out, pos_embeddings_out, ent_embeddings_out], name='concat')
    bilstm_out = Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.25, return_sequences=True), name='bilstm')(
        concatenated_out)

    bilstm_and_depededecies = concatenate([bilstm_out, dep_embeddings_out],
                                          name='bilsm_and_depends')

    gather_layer = Lambda(gather, output_shape=output_shape_of_gather)
    branch_1_input = gather_layer([bilstm_and_depededecies, branch_1_indices])
    branch1_lstm_out = Bidirectional(
        LSTM(300, dropout=0.5, recurrent_dropout=0.25,
             return_sequences=False), name='branch_1_bilstm')(branch_1_input)

    branch_2_input = gather_layer([bilstm_and_depededecies, branch_2_indices])
    branch2_lstm_out = Bidirectional(
        LSTM(300, dropout=0.5, recurrent_dropout=0.25,
             return_sequences=False), name='branch_2_bilstm')(branch_2_input)

    concatenated_branch_out = concatenate([branch1_lstm_out, branch2_lstm_out], name='branch_concat')
    fc_out = Dense(num_classes, activation='softmax')(concatenated_branch_out)

    model = Model([words_indices_input, pos_tag_indices_input, ent_tag_indices_input,
                   dep_tag_indices_input, branch_1_indices, branch_2_indices], fc_out)
    model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy')
    model.name = 'relation_classifier'
    return model
