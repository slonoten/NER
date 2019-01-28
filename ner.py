"""Обучаем и тестируем NER"""

import os

from conll import load_conll
from embeddings import Embeddings
from model import *
from data_generator import DataGenerator
from validation import compute_f1
from ner_model import *

from keras.callbacks import Callback, ModelCheckpoint


class ModelEval(Callback):
    def __init__(self,
                 generator: DataGenerator,
                 labels: List[List[str]],
                 index_to_label: Dict[int, str]):
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


conll_2003_data_dir = './data/conll-2003'

conll_2003_indices = (0, 3)
conll_2003_ignore = ('-DOCSTART-',)

char_cnn_max_token_len = 30

train_data, train_labels = load_conll(os.path.join(conll_2003_data_dir, 'train.txt'),
                                      conll_2003_indices, conll_2003_ignore)
validate_data, validate_labels = load_conll(os.path.join(conll_2003_data_dir, 'dev.txt'),
                                            conll_2003_indices, conll_2003_ignore)
test_data, test_labels = load_conll(os.path.join(conll_2003_data_dir, 'test.txt'),
                                    conll_2003_indices, conll_2003_ignore)

label_classes = sorted({l for sent_labels in train_labels + validate_labels + test_labels for l in sent_labels})

characters = sorted({ch for sent in train_data + validate_data + test_data for word in sent for ch in word})

word_set = {w for sent in train_data + validate_data + test_data for w in sent}

print(f'{len(word_set)} unique words found.')

embed = Embeddings('./embeddings/eng/glove.6B.300d.txt', True, word_set=word_set)
embed_matrix = embed.matrix

train_inputs = make_ner_inputs(train_data, embed, characters, char_cnn_max_token_len)
train_outputs = make_ner_one_hot_outputs(train_labels, label_classes)
validate_inputs = make_ner_inputs(validate_data, embed, characters, char_cnn_max_token_len)
test_inputs = make_ner_inputs(test_data, embed, characters, char_cnn_max_token_len)

# model = build_model_char_cnn_lstm(len(label_classes), embed_matrix, 30, len(characters))
model = build_model_char_cnn_lstm_crf(len(label_classes), embed_matrix, 30, len(characters))

train_generator = DataGenerator(train_inputs, train_outputs, 32)

evaluator = ModelEval(DataGenerator(validate_inputs),
                      validate_labels,
                      label_classes)

model_saver = ModelCheckpoint(filepath='./checkpoints/' + model.name.replace(' ', '_') + '{epoch:02d}.hdf5',
                              verbose=1, save_best_only=False)

# model.fit_generator(train_generator, epochs=50, callbacks=[evaluator, model_saver])

model.load_weights('./checkpoints/char_cnn_bilstm_crf17.hdf5')

prediction = predict(model, DataGenerator(test_inputs), label_classes)

print(compute_f1(prediction, test_labels))

with open('results.txt', 'wt') as file:
    for sent, true_labels, pred_labels in zip(test_data, test_labels, prediction):
        file.writelines([' '.join(z) + '\n' for z in zip(sent, true_labels, pred_labels)])
        file.write('\n')

