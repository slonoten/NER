"""Working with embeddings"""

import numpy as np
from typing import List, Dict, Set
from tqdm import tqdm


class Embeddings:
    def __init__(self,
                 embed_path: str,
                 low_case: bool,
                 vocab_size: int = 0,
                 word_set: Set[str] = None):
        self.low_case = low_case
        if low_case:
            word_set = {w.lower() for w in word_set}
        self.vocab = dict()
        if not vocab_size and not word_set:
            raise ValueError('To load embeddings vocab_size or word_set should be specified.')
        if word_set:
            vocab_size = len(word_set)
        with open(embed_path, 'rt') as embed_file:
            values = embed_file.readline().split(' ')
            if len(values) == 2:
                _, self.embed_dim = map(int, values)
            else:
                self.embed_dim = len(values) - 1
                embed_file.seek(0)
            self.matrix = np.zeros((vocab_size + 2, self.embed_dim))  # zero row is for padding, last row is for OOVW
            token_idx = 1
            all_low_case = True
            for line in tqdm(embed_file):
                values = line.strip().split(' ')
                word = values[0]
                if word_set and word not in word_set:
                    continue
                if all_low_case:
                    all_low_case = all(not ch.isupper() for ch in word)
                self.vocab[word] = token_idx
                self.matrix[token_idx] = np.array(list(map(float, values[1:])))
                token_idx += 1
                if token_idx == vocab_size + 1:
                    break
            if not all_low_case and low_case:
                raise ValueError('Words with upper case symbols found. Check low_case parameter.')
            if all_low_case and not low_case:
                raise ValueError('All words have low case but low_case flag not set.')
            if token_idx != vocab_size + 1:
                self.matrix.resize((token_idx + 1, self.embed_dim), refcheck=False)
        # Out of vocabulary word embedding
        self.oov_word_index = token_idx
        self.matrix[self.oov_word_index] = np.random.normal(size=(1, self.embed_dim))

    def __getitem__(self, word: str) -> int:
        index = self.vocab.get(word.lower() if self.low_case else word)
        return index if index else self.oov_word_index



