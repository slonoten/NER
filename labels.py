"""Работаем с метками"""

from typing import Dict, Iterable, List


def labels_to_indices(label_to_idx: Dict[str, int], labels: Iterable[str]) -> List[int]:
    return [label_to_idx[l] for l in labels]


def indices_to_labels(idx_to_label: Dict[int, str], indices: Iterable[int]) -> List[str]:
    return ['O' if label == 'PAD' else label for label in (idx_to_label[i] for i in indices)]

