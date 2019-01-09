"""Working with embeddings"""

import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm


def load_embeddings(embed_path: str, vocab_size: int, is_fast_text: bool = False) -> Tuple[Dict[str, int], np.array]:
    vocab = dict()
    with open(embed_path, 'rt') as embed_file:
        values = embed_file.readline().split(' ')
        if is_fast_text:
            _, embed_dim = map(int, values)
        else:
            embed_dim = len(values) - 1
            embed_file.seek(0)
        embed_matrix = np.zeros((vocab_size + 2, embed_dim))  # строка 0 для PAD, строка VOCAB_SIZE для OOVW
        token_idx = 1
        for line in tqdm(embed_file):
            values = line.split(' ')
            word = values[0]
            vocab[word] = token_idx
            embed_matrix[token_idx] = np.array(list(map(float, values[1:])))
            token_idx += 1
            if token_idx == vocab_size + 1:
                break

    oov_word_index = token_idx
    embed_matrix[oov_word_index] = np.random.normal(size=(1, embed_dim))  # Эмбеддинг для OOVW
    return vocab, embed_matrix


def sentence_to_indices(vocab: Dict[str, int], words: List[str]) -> List[int]:
    return [(idx if idx else len(vocab) + 1) for idx in (vocab.get(w.lower()) for w in words)]


