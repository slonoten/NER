"""Staff for training NER models"""

from typing import List, Mapping, Any, Dict

import numpy as np

from word_casing import get_casing, CASING_PADDING
from char_features import encode_word


def make_ner_inputs(sentences: List[List[str]],
                    word_to_index: Mapping[str, int],
                    characters: List[str],
                    char_cnn_vector_size: int) -> List[List[List[Any]]]:
    char_to_index = {ch: i + 1 for i, ch in enumerate(characters)}
    inputs = []
    for sentence in sentences:
        word_indexes = [word_to_index[word] for word in sentence]
        word_casings = [get_casing(word) for word in sentence]
        char_encodings = [encode_word(word, char_to_index, char_cnn_vector_size) for word in sentence]
        inputs.append([word_indexes, word_casings, char_encodings])
    padding_values = [0, CASING_PADDING, [0] * char_cnn_vector_size]
    return inputs, padding_values


def build_labels_mapping(labels: List[str]) -> Dict[str, int]:
    return {l: i for i, l in enumerate(['PAD'] + labels)}  # 0 reserved for padding


def to_one_hot(indices: Any, num_classes: int):
    return np.eye(num_classes)[indices]


def make_ner_outputs(sentences_labels: List[List[str]], labels: List[str]) -> List[List[List[int]]]:
    label_to_index = build_labels_mapping(labels)
    return [[[[label_to_index[label]] for label in sentence]] for sentence in sentences_labels], [0]


def make_ner_one_hot_outputs(sentences_labels: List[List[str]], labels: List[str]) -> List[List[List[int]]]:
    label_to_index = build_labels_mapping(labels)
    num_classes = len(label_to_index)
    one_hot_matrix = [[0] * i + [1] + [0] * (num_classes - i - 1) for i in range(num_classes)]
    return [[[one_hot_matrix[label_to_index[label]] for label in sentence]] for sentence in sentences_labels], \
           [one_hot_matrix[0]]

