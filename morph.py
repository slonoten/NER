"""Обучаем и тестируем NER"""

import os
from tqdm import tqdm

from collections import defaultdict, namedtuple

from operator import itemgetter

from conll import load_conll
from embeddings import load_embeddings
from model import *
from data_generator import DataGenerator

from keras.callbacks import Callback, ModelCheckpoint
from typing import List, Tuple, Iterable

TagEncodeInfo = namedtuple('TagEncodeInfo', ('id', 'features'))
TagDecodeInfo = namedtuple('TagDecodeInfo', ('tag', 'features'))


class ModelEval(Callback):
    def __init__(self,
                 generator: DataGenerator,
                 labels: List[List[str]],
                 index_to_label: Dict[int, str],
                 predict_next: bool = False):
        self.labels = labels
        self.predict_next = predict_next
        self.generator = generator
        self.idx_to_label = index_to_label
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        pred = predict(self.model, self.generator, self.idx_to_label)
        f1 = compute_f1(pred, self.labels)
        self.history.append(f1)
        print(f'Precision: {f1[0]}, Recall: {f1[1]}, F1: {f1[2]}')


class FeatureCoder:
    def __init__(self):
        self.pos_tag_to_features = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.tag_to_encode_seq = dict()
        self.tag_to_decode_seq = dict()

    def encode_one(self, tag: str, features: Dict[str, str]):
        code = 0
        if tag == 'ADP':
            print('omg!')
        tag_id, encode_seq = self.tag_to_encode_seq[tag]
        for feature_name, val_dict in encode_seq:
            val = features.get(feature_name)
            i = val_dict[val] if val else 0
            code = code * (len(val_dict) + 1) + i
        return code * len(self.tag_to_encode_seq) + tag_id

    def fit(self, text_pos_tags: Iterable[List[str]], text_features: Iterable[List[str]]) -> None:
        for sent_tags, sent_features in zip(text_pos_tags, tqdm(text_features, 'Analyzing')):
            for tag, features_str in zip(sent_tags, sent_features):
                features = [fp.split('=') for fp in features_str.split('|') if '=' in fp]
                for nvp in features:
                    name, val = nvp
                    self.pos_tag_to_features[tag][name][val] += 1

        for i, (tag, features) in enumerate(self.pos_tag_to_features.items()):
            encode_info = TagEncodeInfo(id=i, features=[(fn, {vn: i + 1 for i, vn in enumerate(fv)})
                                                        for fn, fv in features.items()])
            decode_info = TagDecodeInfo(tag=tag, features=list(reversed([(fn, {i + 1: vn for i, vn in enumerate(fv)})
                                                                        for fn, fv in features.items()])))
            self.tag_to_encode_seq[tag] = encode_info
            self.tag_to_decode_seq[i] = decode_info

    def encode(self, text_pos_tags: Iterable[List[str]], text_features: Iterable[List[str]]) -> List[List[int]]:
        text_morph = []
        for sent_tags, sent_features in zip(text_pos_tags, tqdm(text_features, 'Encoding')):
            sent_morph = []
            for tag, features_str in zip(sent_tags, sent_features):
                features = [fp.split('=') for fp in features_str.split('|') if '=' in fp]
                sent_morph.append((tag, dict(features)))
            text_morph.append([self.encode_one(tag, features) for tag, features in sent_morph])
        return text_morph

    def decode(self, encoded_features: List[List[int]]) -> Tuple[List[List[str]], List[List[str]]]:
        pass


gikrya_data_dir = './data/gikrya'

gikrya_indices = (1, 3, 4)

char_cnn_max_token_len = 30

train_tokens, train_pos, train_features = load_conll(os.path.join(gikrya_data_dir, 'gikrya_new_train.out'),
                                                     gikrya_indices, None)
test_tokens, test_pos, test_features = load_conll(os.path.join(gikrya_data_dir, 'gikrya_new_test.out'),
                                                  gikrya_indices, None)

# label_classes = ['I-PER', 'B-MISC', 'I-MISC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-LOC', 'O', 'B-PER']
pos_tags = sorted({tag for sent_pos_tags in train_pos + test_pos for tag in sent_pos_tags})

feature_encoder = FeatureCoder()
feature_encoder.fit(train_pos + test_pos, train_features + test_features)
train_morph = list(feature_encoder.encode(train_pos, train_features))
test_morph = (feature_encoder.encode(test_pos, test_features))


characters = sorted({ch for sent in train_tokens + test_tokens for word in sent for ch in word})
# characters = """!"#$%&\\'()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]`abcdefghijklmnopqrstuvwxyz"""
char_to_idx = {ch: i + 1 for i, ch in enumerate(characters)}

vocab, embed_matrix = load_embeddings('./embeddings/glove.6B.300d.txt', 40000)

# vocab, embed_matrix = load_embeddings('/home/max/ipython/sber-nlp-course/wiki-news-300d-1M-subword.vec', 40000)

embed_dim = embed_matrix.shape[1]

model = build_model_char_cnn_lstm(len(label_classes), embed_matrix, 30, len(characters))

train_generator = DataGenerator(train_data, vocab, char_to_idx, train_labels,
                                label_to_idx, 32, max_token_len=char_cnn_max_token_len)

evaluator = ModelEval(DataGenerator(validate_data, vocab, char_to_idx, max_token_len=char_cnn_max_token_len),
                      validate_labels,
                      idx_to_label)

model_saver = ModelCheckpoint(filepath='./checkpoints/crf_model_x{epoch:02d}.hdf5', verbose=1, save_best_only=False)

model.fit_generator(train_generator, epochs=50, steps_per_epoch=len(train_generator),
                    callbacks=[evaluator, model_saver])

#  model.load_weights('./checkpoints/model_v3_29.hdf5')

prediction = predict(model,
                     DataGenerator(test_data, vocab, char_to_idx, max_token_len=char_cnn_max_token_len),
                     idx_to_label)

print(compute_f1(prediction, test_labels))

with open('results.txt', 'wt') as file:
    for sent, true_labels, pred_labels in zip(test_data, test_labels, prediction):
        file.writelines([' '.join(z) + '\n' for z in zip(sent, true_labels, pred_labels)])
        file.write('\n')
