"""Готовим данные для модели"""

import numpy as np

from typing import List, Dict, Any
from operator import itemgetter
from random import sample
from itertools import groupby, accumulate

from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

from embeddings import sentence_to_indices
from word_casing import get_casing, CASING_PADDING


def pad_sequence(sequence: List[Any], max_len: int, value: Any):
    return ([value] * max(0, max_len - len(sequence)) + sequence)[:max_len]


def word_to_char_indices(char_to_index: Dict[str, int], word: str, max_len: int):
    return pad_sequence([c if c else char_to_index['*'] for c in (char_to_index.get(ch) for ch in word)], max_len, 0)


def to_categorical(indices: Any, num_classes: int):
    return np.eye(num_classes)[indices]


class DataGenerator(Sequence):
    @staticmethod
    def shuffle_indices(sorted_lengths, num_samples):
        shuffled_by_len = [(i, l)
                           for y in (list(x) for _, x in groupby(sorted_lengths, lambda x: x[1] // 10))
                           for i, l in sample(y, len(y))]
        indices = sample(range(len(shuffled_by_len)), num_samples)
        return [shuffled_by_len[i] for i in indices]

    def __init__(self, sentences: List[List[str]],
                 vocab: Dict[str, int],
                 char_to_idx: Dict[str, int],
                 labels: List[List[str]] = None,
                 label_to_idx: Dict[str, int] = None,
                 batch_size: int = -1,
                 predict_next: bool = False,):

        self.predict = not labels
        self.predict_next = predict_next
        lengths = ((i, len(sent)) for i, sent in enumerate(sentences))
        self.sorted_lengths = sorted(lengths, key=itemgetter(1))
        if self.predict:
            self.indices = self.sorted_lengths
            range_lengths = [len(list(g)) for _, g in groupby(self.sorted_lengths, itemgetter(1))]
            starts = list(accumulate(range_lengths))
            self.ranges = list(zip([0] + starts, starts))
        else:
            num_batches = len(sentences) // batch_size
            self.indices = DataGenerator.shuffle_indices(self.sorted_lengths, num_batches * batch_size)
            self.ranges = [(i * batch_size, (i + 1) * batch_size) for i in range(num_batches)]
            # Подготовим метки
            self.num_classes = len(label_to_idx)
            self.labels = [[label_to_idx[l] for l in sent] for sent in labels]
            self.label_pad_value = label_to_idx['O']
        self.word_indices = [sentence_to_indices(vocab, sent) for sent in sentences]
        self.word_casing = [[get_casing(w) for w in sent] for sent in sentences]
        self.word_characters = [[word_to_char_indices(char_to_idx, w, 32) for w in sent] for sent in sentences]

    def __getitem__(self, index: int):
        start, end = self.ranges[index]
        indices_and_len_tuples = self.indices[start:end]
        pad_to_len = max(l for _, l in indices_and_len_tuples)
        indices = [i for i, _ in indices_and_len_tuples]
        words_input = pad_sequences([self.word_indices[i] for i in indices], pad_to_len)
        casing_input = np.array([pad_sequence(self.word_casing[i], pad_to_len, CASING_PADDING) for i in indices])
        characters_input = np.array([pad_sequence(self.word_characters[i], pad_to_len, [0] * 32) for i in indices])
        if self.predict:
            return [words_input, casing_input, characters_input]
        labels = [self.labels[i] for i in indices]
        labels_cat = to_categorical(pad_sequences(labels, pad_to_len, value=self.label_pad_value),
                                    self.num_classes)
        if self.predict_next:
            labels_next = [sent_labels[1:] + [self.label_pad_value] for sent_labels in labels]
            labels_next_cat = to_categorical(pad_sequences(labels_next, pad_to_len, value=self.label_pad_value),
                                    self.num_classes)
            labels_prev = [[self.label_pad_value] + sent_labels[:-1] for sent_labels in labels]
            labels_prev_cat = to_categorical(pad_sequences(labels_prev, pad_to_len, value=self.label_pad_value),
                                    self.num_classes)
            return [words_input, casing_input, characters_input], [labels_cat, labels_prev_cat, labels_next_cat]

        return [words_input, casing_input, characters_input], labels_cat

    def __len__(self) -> int:
        return len(self.ranges)

    def on_epoch_end(self):
        self.indices = DataGenerator.shuffle_indices(self.sorted_lengths, len(self.indices))

    def get_indices_and_lengths(self, index: int):
        start, end = self.ranges[index]
        return self.indices[start:end]