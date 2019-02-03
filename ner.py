"""Обучаем и тестируем NER"""

import os
from typing import Iterable, List
from datetime import datetime

from conll import load_conll
from embeddings import Embeddings
from model import *
from data_generator import DataGenerator
from validation import compute_f1
from ner_model import make_ner_inputs, make_ner_one_hot_outputs, build_model_char_cnn_lstm_crf
from labels import transform_indices_to_labels, build_labels_mapping, build_indices_mapping

from keras.callbacks import Callback, ModelCheckpoint, CSVLogger


class ModelEval(Callback):
    def __init__(self,
                 generator: DataGenerator,
                 valid_labels: Iterable[List[str]],
                 index_to_label: Dict[int, str]):
        super(ModelEval, self).__init__()
        self.valid_labels = list(valid_labels)
        self.index_to_label = index_to_label
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred = transform_indices_to_labels(self.index_to_label, predict(self.model, self.generator))
        f1 = compute_f1(pred, self.valid_labels)
        logs['valid_f1'] = f1[0]
        print(f'Precision: {f1[0]}, Recall: {f1[1]}, F1: {f1[2]}')


def main():
    conll_2003_data_dir = './data/conll-2003'
    conll_2003_indices = [0, 3]
    conll_2003_ignore = ('-DOCSTART-',)

    char_cnn_width = 30

    train_data, train_labels = load_conll(os.path.join(conll_2003_data_dir, 'train.txt'),
                                          conll_2003_indices, conll_2003_ignore)
    validate_data, validate_labels = load_conll(os.path.join(conll_2003_data_dir, 'dev.txt'),
                                                conll_2003_indices, conll_2003_ignore)
    test_data, test_labels = load_conll(os.path.join(conll_2003_data_dir, 'test.txt'),
                                        conll_2003_indices, conll_2003_ignore)

    label_classes = sorted({l for sent_labels in train_labels + validate_labels + test_labels for l in sent_labels})
    label_to_index = build_labels_mapping(label_classes)
    index_to_label = build_indices_mapping(label_classes)

    characters = sorted({ch for sent in train_data + validate_data + test_data for word in sent for ch in word})

    word_set = {w for sent in train_data + validate_data + test_data for w in sent}

    print(f'{len(word_set)} unique words found.')

    embed = Embeddings('./embeddings/eng/glove.6B.300d.txt', True, word_set=word_set)
    embed_matrix = embed.matrix

    train_inputs = make_ner_inputs(train_data, embed, characters, char_cnn_width)
    train_outputs = make_ner_one_hot_outputs(train_labels, label_to_index)
    validate_inputs = make_ner_inputs(validate_data, embed, characters, char_cnn_width)
    test_inputs = make_ner_inputs(test_data, embed, characters, char_cnn_width)

    model = build_model_char_cnn_lstm_crf(len(label_classes), embed_matrix, 30, len(characters))

    train_generator = DataGenerator(train_inputs, train_outputs, 32)

    evaluator = ModelEval(DataGenerator(validate_inputs),
                          validate_labels,
                          index_to_label)

    model_saver = ModelCheckpoint(filepath='./checkpoints/' + model.name.replace(' ', '_') + '_{epoch:02d}.hdf5',
                                  verbose=1, save_best_only=True, monitor='valid_f1')

    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    csv_logger = CSVLogger(f"NER_log_{time_stamp}.csv", append=False)

    # model.load_weights('./checkpoints/char_cnn_bilstm_crf17.hdf5')

    model.fit_generator(train_generator, epochs=1, callbacks=[evaluator, model_saver, csv_logger])

    test_pred_indices = predict(model, DataGenerator(test_inputs))

    test_pred_labels = transform_indices_to_labels(index_to_label, test_pred_indices)

    print(compute_f1(test_pred_labels, test_labels))

    with open('ner_results.txt', 'wt') as file:
        for sent, sent_true_labels, sent_pred_labels in zip(test_data, test_labels, test_pred_labels):
            file.writelines([' '.join(z) + '\n' for z in zip(sent, sent_true_labels, sent_pred_labels)])
            file.write('\n')


if __name__ == '__main__':
    main()
