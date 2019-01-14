"""Дополнение последовательности до заданной длины"""
from typing import List, Any


def pad_sequence_left(sequence: List[Any], max_len: int, pad_value: Any):
    return ([pad_value] * max(0, max_len - len(sequence)) + sequence)[:max_len]


def pad_sequence_right(sequence: List[Any], max_len: int, pad_value: Any):
    num_to_pad = max_len - len(sequence)
    return (sequence + [pad_value] * max(0, num_to_pad))[-min(num_to_pad, 0):]


def pad_sequence_center(sequence: List[Any], max_len: int, pad_value: Any):
    max_len_left = max_len // 2
    max_len_right = max_len - max_len_left
    seq_len = len(sequence)
    return pad_sequence_left(sequence[:seq_len // 2], max_len_left, pad_value)\
        + pad_sequence_right(sequence[seq_len // 2:], max_len_right, pad_value)
