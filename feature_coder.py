"""Class for encoding-decoding morpho features and tags"""

import operator

from functools import reduce
from typing import List, Tuple, Iterable, Dict
from tqdm import tqdm
from collections import namedtuple, defaultdict


TagEncodeInfo = namedtuple('TagEncodeInfo', ('id', 'features'))
TagDecodeInfo = namedtuple('TagDecodeInfo', ('tag', 'features'))


class FeatureCoder:
    def __init__(self):
        self.pos_tag_to_features = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.tag_to_encode_seq = None
        self.tag_id_to_decode_seq = None
        self.tag_ranges = None
        self.tag_ranges_rev = None
        self.padding_shift = 1  # reserve 0 for padding

    def decode_one(self, code: int) -> Tuple[str, List[Tuple[str, str]]]:
        assert self.tag_ranges_rev, 'Coder not initialized'
        for tag_id, start_range in self.tag_ranges_rev:
            if code >= start_range:
                code -= start_range
                break
        tag, decode_seq = self.tag_id_to_decode_seq[tag_id]
        features = []
        for feature_name, val_id_to_value in decode_seq:
            num_values = len(val_id_to_value) + 1  # + 1 feature not set case
            val_id = code % num_values
            code = code // num_values
            if val_id:
                features.append((feature_name, val_id_to_value[val_id]))
        return tag, features

    def encode_one(self, tag: str, features: Dict[str, str]) -> int:
        code = 0
        tag_id, encode_seq = self.tag_to_encode_seq[tag]
        for feature_name, val_dict in encode_seq:
            val = features.get(feature_name)
            i = val_dict[val] if val else 0
            code = code * (len(val_dict) + 1) + i
        code += self.tag_ranges[tag_id]
        return code

    def fit(self, text_pos_tags: Iterable[List[str]], text_features: Iterable[List[str]]) -> int:
        for sent_tags, sent_features in zip(text_pos_tags, tqdm(text_features, 'Analyzing features')):
            for tag, features_str in zip(sent_tags, sent_features):
                features = [fp.split('=') for fp in features_str.split('|') if '=' in fp]
                if features:
                    for nvp in features:
                        name, val = nvp
                        self.pos_tag_to_features[tag][name][val] += 1
                else:
                    self.pos_tag_to_features[tag]
        num_classes = self.padding_shift  # 0 class reserved for padding
        self.tag_to_encode_seq = dict()
        self.tag_id_to_decode_seq = []
        self.tag_ranges = []
        for i, (tag, features) in enumerate(self.pos_tag_to_features.items()):
            encode_info = TagEncodeInfo(id=i, features=[(fn, {vn: i + 1 for i, vn in enumerate(fv)})
                                                        for fn, fv in features.items()])
            decode_info = TagDecodeInfo(tag=tag, features=list(reversed([(fn, {i + 1: vn for i, vn in enumerate(fv)})
                                                                        for fn, fv in features.items()])))
            self.tag_to_encode_seq[tag] = encode_info
            self.tag_id_to_decode_seq.append(decode_info)
            self.tag_ranges.append(num_classes)
            num_classes += reduce(operator.mul, (len(f[1]) + 1 for f in encode_info.features), 1)
        self.tag_ranges_rev = list(reversed(list(enumerate(self.tag_ranges))))
        return num_classes

    def encode(self, text_pos_tags: Iterable[List[str]], text_features: Iterable[List[str]]) -> List[List[int]]:
        text_morph = []
        for sent_tags, sent_features in zip(text_pos_tags, tqdm(text_features, 'Encoding features')):
            sent_morph = []
            for tag, features_str in zip(sent_tags, sent_features):
                features = [fp.split('=') for fp in features_str.split('|') if '=' in fp]
                sent_morph.append((tag, dict(features)))
            text_morph.append([self.encode_one(tag, features) for tag, features in sent_morph])
        return text_morph

    def decode(self, encoded_features: List[List[int]]) -> Tuple[List[List[str]], List[List[str]]]:
        text_pos, text_features = [], []
        for sent_codes in tqdm(encoded_features, 'Decoding features'):
            sent_pos, sent_features = [], []
            for code in sent_codes:
                pos, feature_list = self.decode_one(code)
                sent_pos.append(pos)
                sent_features.append('|'.join(sorted(f'{k}={v}' for k, v in feature_list)) if feature_list else '_')
            text_pos.append(sent_pos)
            text_features.append(sent_features)
        return text_pos, text_features
