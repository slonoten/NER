"""Embeddings based on frequences of appearing of word morphological features"""

import numpy as np

from typing import Iterable, List
from collections import defaultdict
from tqdm import tqdm


class MorphoEmbeddings:
    def __init__(self):
        self.embed_matrix = None
        self.word_to_index = dict()
        self.oow_index = -1

    def fit(self,
            text: Iterable[List[str]],
            text_pos_tags: Iterable[List[str]],
            text_features: Iterable[List[str]]) -> None:
        self.word_to_index.clear()
        word_counts = defaultdict(lambda: 0)
        word_pos_counts = defaultdict(lambda: defaultdict(lambda: 0))
        word_features_counts = defaultdict(lambda: defaultdict(lambda: 0))
        pos_tags_counts, feature_counts = defaultdict(lambda: 0), defaultdict(lambda: 0)
        for sentence, sent_pos_tags, sent_features in tqdm(zip(text, text_pos_tags, text_features)):
            for word, pos_tag, features in zip(sentence, sent_pos_tags, sent_features):
                word = word.lower()
                pos_tags_counts[pos_tag] += 1
                word_counts[word] += 1
                word_pos_counts[word][pos_tag] += 1
                current_word_feature_counts = word_features_counts[word]
                for feature in features.split('|'):
                    current_word_feature_counts[feature] += 1
                    feature_counts[feature] += 1
        num_words = len(word_counts)
        num_pos = len(pos_tags_counts)
        embed_dim = num_pos + len(feature_counts)
        self.embed_matrix = np.zeros((num_words + 2, embed_dim))
        pos_to_index = {pos: i for i, pos in enumerate(pos_tags_counts.keys())}
        feature_to_index = {feature: i + num_pos for i, feature in enumerate(feature_counts.keys())}
        for i, (word, word_count) in enumerate(word_counts.items()):
            word_embedding = self.embed_matrix[i + 1]  # 0 index reserved for padding
            for feature, feature_count in word_features_counts[word].items():
                word_embedding[feature_to_index[feature]] = feature_count / word_count
            for pos, pos_count in word_pos_counts[word].items():
                word_embedding[pos_to_index[pos]] = pos_count / word_count
            self.word_to_index[word] = i + 1
        # out of vocabulary embedding
        self.oow_index = num_words + 1
        oow_embedding = self.embed_matrix[self.oow_index]
        for feature, feature_count in feature_counts.items():
            oow_embedding[feature_to_index[feature]] = feature_count / num_words
        for pos, pos_count in pos_tags_counts.items():
            word_embedding[pos_to_index[pos]] = pos_count / num_words

    def __getitem__(self, word: str) -> int:
        index = self.word_to_index.get(word)
        return index if index else self.oow_index



