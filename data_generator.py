"""Готовим данные для модели"""

import numpy as np

from typing import List, Dict, Any, Union, Mapping, Iterable, Tuple
from operator import itemgetter
from random import sample
from itertools import groupby, accumulate

from keras.utils import Sequence
from padding import pad_sequence_left


class DataGenerator(Sequence):
    @staticmethod
    def shuffle_indices(sorted_lengths, num_samples):
        shuffled_by_len = [(i, l)
                           for y in (list(x) for _, x in groupby(sorted_lengths, lambda x: x[1] // 10))
                           for i, l in sample(y, len(y))]
        indices = sample(range(len(shuffled_by_len)), num_samples)
        return [shuffled_by_len[i] for i in indices]

    def __init__(self,
                 input_data: Tuple[List[List[List[Any]]], List[Any]],
                 output_data: Tuple[List[List[List[Any]]], List[Any]] = None,
                 batch_size: int = 32):
        self.inputs, self.input_pad_values = input_data
        self.num_inputs = len(self.inputs[0])
        lengths = ((i, len(input[0])) for i, input in enumerate(self.inputs))
        self.sorted_lengths = sorted(lengths, key=itemgetter(1))
        self.predict = not output_data
        if self.predict:
            self.indices = self.sorted_lengths
            range_lengths = [len(list(g)) for _, g in groupby(self.sorted_lengths, itemgetter(1))]
            starts = list(accumulate(range_lengths))
            self.ranges = list(zip([0] + starts, starts))
        else:
            num_batches = len(self.inputs) // batch_size
            self.indices = DataGenerator.shuffle_indices(self.sorted_lengths, num_batches * batch_size)
            self.ranges = [(i * batch_size, (i + 1) * batch_size) for i in range(num_batches)]
            self.outputs, self.outputs_pad_values = output_data
            self.num_outputs = len(self.outputs[0])

    def __getitem__(self, index: int):
        start, end = self.ranges[index]
        indices_and_len_tuples = self.indices[start:end]
        pad_to_len = max(l for _, l in indices_and_len_tuples)
        indices = [i for i, _ in indices_and_len_tuples]
        batch_inputs = [[] for _ in range(self.num_inputs)]
        for i in indices:
            for j in range(self.num_inputs):
                batch_inputs[j].append(pad_sequence_left(self.inputs[i][j], pad_to_len, self.input_pad_values[j]))
        input_tensors = [np.array(bi) for bi in batch_inputs]
        if self.predict:
            return input_tensors

        batch_outputs = [[] for _ in range(self.num_outputs)]
        for i in indices:
            for j in range(self.num_outputs):
                batch_outputs[j].append(
                    pad_sequence_left(self.outputs[i][j], pad_to_len, self.outputs_pad_values[j]))
        output_tensors = [np.array(bo) for bo in batch_outputs]
        return input_tensors, output_tensors

    def __len__(self) -> int:
        num_batches = len(self.ranges)
        assert num_batches, 'Number of batches should be greater then zero.'
        return num_batches

    def on_epoch_end(self):
        self.indices = DataGenerator.shuffle_indices(self.sorted_lengths, len(self.indices))

    def get_indices_and_lengths(self, index: int):
        start, end = self.ranges[index]
        return self.indices[start:end]
