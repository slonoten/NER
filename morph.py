"""Обучаем и тестируем NER"""

import os
from datetime import datetime

from conll import load_conll
from morpho_embeddings import MorphoEmbeddings
from morpho_model import make_morpho_inputs, make_morpho_outputs, build_model_morph_rnn
from model import predict

from feature_coder import FeatureCoder
from data_generator import DataGenerator

from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from typing import List, Any

from sklearn.metrics import f1_score, precision_score, recall_score


def flatten(lst: List[List[Any]]) -> List[Any]:
    return [x for inner in lst for x in inner]


def text_to_lower(text: List[List[str]]) -> List[List[str]]:
    return [[t.lower() for t in sent] for sent in text]


class ModelEval(Callback):
    def __init__(self,
                 generator: DataGenerator,
                 labels: List[List[int]]):
        super(ModelEval, self).__init__()
        self.labels_flat = flatten(labels)
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred_flat = flatten(predict(self.model, self.generator))
        precission = precision_score(pred_flat, self.labels_flat, average='weighted')
        recall = recall_score(pred_flat, self.labels_flat, average='weighted')
        f1 = 2 * precission * recall / (precission + recall)
        logs['valid_f1'] = f1
        print(f'Precision: {precission}, Recall: {recall}, F1: {f1}')


gikrya_data_dir = './data/gikrya'
gikrya_indices = [1, 3, 4]
char_cnn_width = 30
train_tokens, train_pos, train_features = load_conll(os.path.join(gikrya_data_dir, 'gikrya_new_train.out'),
                                                     gikrya_indices, None)
test_tokens, test_pos, test_features = load_conll(os.path.join(gikrya_data_dir, 'gikrya_new_test.out'),
                                                  gikrya_indices, None)

morpho_embeddings = MorphoEmbeddings()

morpho_embeddings.fit(train_tokens, train_pos, train_features)

feature_encoder = FeatureCoder()
num_classes = feature_encoder.fit(train_pos + test_pos, train_features + test_features)

# checking encode-decode result
test_morph = list(feature_encoder.encode(test_pos, test_features))
check_pos, check_features = feature_encoder.decode(test_morph)
assert check_pos == test_pos
assert check_features == test_features

characters = sorted({ch for sent in train_tokens + test_tokens for word in sent for ch in word})
char_to_idx = {ch: i + 1 for i, ch in enumerate(characters)}

embed_matrix = morpho_embeddings.embed_matrix
embed_dim, _ = embed_matrix.shape

model = build_model_morph_rnn(num_classes, embed_matrix, 30, len(characters))

train_inputs = make_morpho_inputs(train_tokens, morpho_embeddings, characters, char_cnn_width)
train_outputs = make_morpho_outputs(train_pos, train_features, feature_encoder)

train_generator = DataGenerator(train_inputs, train_outputs)

test_inputs = make_morpho_inputs(test_tokens, morpho_embeddings, characters, char_cnn_width)
evaluator = ModelEval(DataGenerator(test_inputs),
                      test_morph)

model_saver = ModelCheckpoint(filepath='./checkpoints/' + model.name.replace(' ', '_') + '_{epoch:02d}.hdf5',
                              verbose=1, save_best_only=False)

time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
csv_logger = CSVLogger(f"./logs/Morph_log_{time_stamp}.csv", append=False)

# model.load_weights('./checkpoints/morph_01.hdf5')
model.fit_generator(train_generator, epochs=50, steps_per_epoch=len(train_generator),
                    callbacks=[model_saver, evaluator, csv_logger])

prediction = predict(model, DataGenerator(test_inputs))

print(f1_score(flatten(prediction), flatten(test_morph), average='weighted'))
