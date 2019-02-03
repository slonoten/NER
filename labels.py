"""Работаем с метками"""

from typing import Dict, Iterable, List


def build_labels_mapping(labels: List[str]) -> Dict[str, int]:
    return {l: i for i, l in enumerate(['PAD'] + labels)}  # 0 reserved for padding


def build_indices_mapping(labels: List[str]) -> Dict[int, str]:
    return {i: l for i, l in enumerate(['PAD'] + labels)}  # 0 reserved for padding


def transform_labels_to_indices(label_to_idx: Dict[str, int], labels: Iterable[List[str]]) -> List[List[int]]:
    return [[label_to_idx[l] for l in sent] for sent in labels]


def transform_indices_to_labels(idx_to_label: Dict[int, str],
                                indices: Iterable[List[int]],
                                pad_label: str = 'O') -> List[List[str]]:
    return [[pad_label if label == 'PAD' else label for label in (idx_to_label[i] for i in sent)] for sent in indices]

