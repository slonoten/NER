"""Staff for training NER models"""

from typing import List, Mapping, Any, Tuple


from word_casing import get_casing, CASING_PADDING
from char_features import encode_word
from feature_coder import FeatureCoder


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
