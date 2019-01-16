"""Обучаем и тестируем NER"""

import os

from conll import load_conll
from embeddings import load_embeddings
from model import *
from data_generator import DataGenerator
from validation import compute_f1

from keras.callbacks import Callback, ModelCheckpoint


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


conll_2003_data_dir = './data/conll-2003'

conll_2003_indices = (0, 3)
conll_2003_ignore = ('-DOCSTART-',)

char_cnn_max_token_len = 30

train_data, train_labels = load_conll(os.path.join(conll_2003_data_dir, 'train.txt'),
                                      conll_2003_indices, conll_2003_ignore)
validate_data, validate_labels = load_conll(os.path.join(conll_2003_data_dir, 'valid.txt'),
                                            conll_2003_indices, conll_2003_ignore)
test_data, test_labels = load_conll(os.path.join(conll_2003_data_dir, 'test.txt'),
                                    conll_2003_indices, conll_2003_ignore)

# label_classes = ['I-PER', 'B-MISC', 'I-MISC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-LOC', 'O', 'B-PER']
label_classes = sorted({l for sent_labels in train_labels + validate_labels + test_labels for l in sent_labels})
label_to_idx = {l: i for i, l in enumerate(label_classes)}
idx_to_label = {i: l for l, i in label_to_idx.items()}

characters = sorted({ch for sent in train_data + validate_data + test_data for word in sent for ch in word})
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

