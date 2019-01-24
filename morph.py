"""Обучаем и тестируем NER"""

import os

from conll import load_conll
from embeddings import load_embeddings
from model import *

from feature_coder import FeatureCoder
from data_generator import DataGenerator

from keras.callbacks import Callback, ModelCheckpoint
from typing import List, Any, Tuple, Iterable

from sklearn.metrics import f1_score, precision_score, recall_score


def flatten(lst: List[List[Any]]) -> List[Any]:
    return [x for inner in lst for x in inner]


def text_to_lower(text: List[List[str]]) -> List[List[str]]:
    return [[t.lower() for t in sent] for sent in text]


class ModelEval(Callback):
    def __init__(self,
                 generator: DataGenerator,
                 labels: List[List[int]]):
        self.labels_flat = flatten(labels)
        self.generator = generator
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        pred_flat = flatten(predict(self.model, self.generator))
        prec = precision_score(pred_flat, self.labels_flat, average='weighted')
        recall = recall_score(pred_flat, self.labels_flat, average='weighted')
        f1 = 2 * prec * recall / (prec + recall)
        self.history.append(f1)
        print(f'Precision: {prec}, Recall: {recall}, F1: {f1}')


gikrya_data_dir = './data/gikrya'

gikrya_indices = (1, 3, 4)

char_cnn_max_token_len = 30

train_tokens, train_pos, train_features = load_conll(os.path.join(gikrya_data_dir, 'gikrya_new_train.out'),
                                                     gikrya_indices, None)
test_tokens, test_pos, test_features = load_conll(os.path.join(gikrya_data_dir, 'gikrya_new_test.out'),
                                                  gikrya_indices, None)

pos_tags = sorted({tag for sent_pos_tags in train_pos + test_pos for tag in sent_pos_tags})

feature_encoder = FeatureCoder()
num_classes = feature_encoder.fit(train_pos + test_pos, train_features + test_features)
train_morph = list(feature_encoder.encode(train_pos, train_features))
test_morph = list(feature_encoder.encode(test_pos, test_features))

check_pos, check_features = feature_encoder.decode(test_morph)
assert check_pos == test_pos
assert check_features == test_features

characters = sorted({ch for sent in train_tokens + test_tokens for word in sent for ch in word})
char_to_idx = {ch: i + 1 for i, ch in enumerate(characters)}

token_set = {t.lower() for sent in train_tokens + test_tokens for t in sent}

vocab, embed_matrix = load_embeddings('./embeddings/rus/ft_native_300_ru_wiki_lenta_lemmatize.vec',
                                      word_set=token_set,
                                      is_fast_text=True)
train_tokens, test_tokens = text_to_lower(train_tokens), text_to_lower(test_tokens)

embed_dim, _ = embed_matrix.shape

model = build_model_predict_neighbour(num_classes, embed_matrix, 30, len(characters))

train_pred_left = list(map(lambda s: [0] + s[:-1], train_morph))
train_pred_right = list(map(lambda s: s[1:] + [0], train_morph))

train_generator = DataGenerator(train_tokens, vocab, char_to_idx,
                                categorical_labels=[train_morph, train_pred_left, train_pred_right],
                                batch_size=32, max_token_len=char_cnn_max_token_len)

evaluator = ModelEval(DataGenerator(test_tokens, vocab, char_to_idx,
                                    max_token_len=char_cnn_max_token_len),
                      test_morph)

model_saver = ModelCheckpoint(filepath='./checkpoints/morph_multi_loss_{epoch:02d}.hdf5', verbose=1, save_best_only=False)

model.fit_generator(train_generator, epochs=50, steps_per_epoch=len(train_generator),
                    callbacks=[model_saver, evaluator])

# model.load_weights('./checkpoints/morph_01.hdf5')

prediction = predict(model, DataGenerator(test_tokens, vocab, char_to_idx, max_token_len=char_cnn_max_token_len))

print(f1_score(flatten(prediction), flatten(test_morph), average='weighted'))
