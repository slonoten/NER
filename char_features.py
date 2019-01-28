"""Character level features"""

from typing import List, Mapping

from padding import pad_sequence_center


def encode_word(word: str, char_to_index: Mapping[str, int], encoding_len: int) -> List[int]:
    return pad_sequence_center([char_to_index[ch] for ch in word], encoding_len, 0)
