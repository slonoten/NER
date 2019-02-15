"""Обучаем и тестируем NER"""

import os
from typing import Iterable, List, Tuple, Dict
from datetime import datetime

from conll import load_conll
from embeddings import Embeddings
from model import predict
from data_generator import DataGenerator
from validation import compute_f1
from rel_ext_model import make_rel_ext_inputs, build_rel_ext_model
from ner_model import make_ner_outputs
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


def get_entity_positions(sentence: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    start_pos = sentence.find('"') + 1
    assert start_pos >= 0, 'Sentence must be quoted'
    e1_pos = sentence.find('<e1>') - start_pos, sentence.find('</e1>') - start_pos - len('<e1>')
    e2_pos = sentence.find('<e2>') - start_pos - len('<e1></e1>'), \
        sentence.find('</e2>') - start_pos - len('<e1></e1><e2>')
    assert e1_pos[0] >= 0, "Position not found"
    assert e1_pos[1] >= 0, "Position not found"
    assert e2_pos[0] >= 0, "Position not found"
    assert e2_pos[1] >= 0, "Position not found"
    return e1_pos, e2_pos


def get_range_index(rng: Tuple[int, int], tokens: Iterable[Tuple[int, int]]) -> int:
    for index, token in enumerate(tokens):
        if token[0] == rng[0] and token[1] == rng[1]:
            return index
    assert False, "Range index not found"


def get_path_to_root(depend_ptrs: List[int], index: int) -> Iterable[int]:
    while True:
        if 0 > depend_ptrs[index] >= len(depend_ptrs):
            raise ValueError("Invalid pointer")
        yield index
        if depend_ptrs[index] <= 0:
            break
        index = depend_ptrs[index] - 1


def get_tree_path(depend_ptrs: List[int], idx1: int, idx2: int) -> Tuple[List[int], List[int]]:
    path1 = list(get_path_to_root(depend_ptrs, idx1))
    path2 = list(get_path_to_root(depend_ptrs, idx2))
    if path1[-1] != path2[-1]:
        print(f'Bad dependencies: {depend_ptrs}')
        return path1, path2  # Something goes wrong with dependencies
    i = 0
    while True:
        if min(len(path1), len(path2)) - 1 == i:
            break
        if path1[-2 - i] != path2[-2 - i]:
            break
        i += 1
    return (path1[:-i], path2[:-i]) if i > 0 else (path1, path2)


def load_relations(file_name: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    labels = []
    pair_positions = []
    with open(file_name, 'rt') as file:
        while True:
            sentence = file.readline()
            if not sentence:
                break
            e1_range, e2_range = get_entity_positions(sentence)
            e1, e2 = e1_range[0], e2_range[0]
            label_line = file.readline()
            bracket_pos = label_line.find('(e')
            if bracket_pos < 0:
                labels.append(label_line.rstrip())
                pair_positions.append((e1, e2))
            else:
                labels.append(label_line[:bracket_pos])
                if label_line[bracket_pos + 2] == '1':
                    pair_positions.append((e1, e2))
                else:
                    pair_positions.append((e2, e1))
            file.readline()  # Comment line
            file.readline()  # Delimiter line
    return labels, pair_positions


def find_nearest_index(positions: List[int], position) -> int:
    candidates = [position, position + 1, position - 1, position + 2, position - 2]
    for p in candidates:
        try:
            return positions.index(p)
        except ValueError:
            pass
    raise Exception("Position not found")


def build_branches_indices(leafs_positions: Iterable[Tuple[int, int]],
                           tokens_positions: List[List[int]],
                           depends: List[List[int]]) -> Tuple[List[List[int]]]:
    indices1, indices2 = [], []
    for i, ((p1, p2), sent_positions, sent_depends) in enumerate(zip(leafs_positions, tokens_positions, depends)):
        if i == 1768:
            print(i)
        i1, i2 = find_nearest_index(sent_positions, p1), find_nearest_index(sent_positions, p2)
        path1, path2 = get_tree_path(sent_depends, i1, i2)
        indices1.append(path1)
        indices2.append(path2)
    return indices1, indices2


def str_to_int(lst: List[List[str]]):
    return [list(map(int, sent)) for sent in lst]


def main():
    sem_eval_data_dir = './data/semeval-2010-task-8'
    sem_eval_indices = [0, 1, 3, 5, 6, 7]

    train_words, train_starts, train_pos, train_link, train_dep, train_ent_labels = \
        load_conll(os.path.join(sem_eval_data_dir, 'TRAIN_FILE.TXT.all'), sem_eval_indices)

    train_starts = str_to_int(train_starts)
    train_link = str_to_int(train_link)

    train_rel_labels, train_pair_positions = load_relations(os.path.join(sem_eval_data_dir,
                                                                         'TRAIN_FILE.TXT'))

    train_branch1, train_branch2 = build_branches_indices(train_pair_positions, train_starts, train_link)

    test_words, test_starts, test_pos, test_link, test_dep, test_ent_labels = \
        load_conll(os.path.join(sem_eval_data_dir, 'TEST_FILE_FULL.TXT.all'), sem_eval_indices)

    test_starts = str_to_int(test_starts)
    test_link = str_to_int(test_link)

    test_rel_labels, test_pair_positions = load_relations(os.path.join(sem_eval_data_dir,
                                                                       'TEST_FILE_FULL.TXT'))
    test_branch1, test_branch2 = build_branches_indices(test_pair_positions, test_starts, test_link)

    pos_classes = sorted({l for sent_pos in train_pos + test_pos for l in sent_pos})
    pos_to_index = build_labels_mapping(pos_classes)

    label_classes = sorted({l for sent_labels in train_ent_labels + test_ent_labels for l in sent_labels})
    label_to_index = build_labels_mapping(label_classes)
    index_to_label = build_indices_mapping(label_classes)

    dep_classes = sorted({l for sent_dep in train_dep + test_dep for l in sent_dep})
    dep_to_index = build_labels_mapping(dep_classes)

    word_set = {w for sent in train_words + test_words for w in sent}

    print(f'{len(word_set)} unique words found.')

    embed = Embeddings('./embeddings/eng/glove.6B.300d.txt', True, word_set=word_set)
    embed_matrix = embed.matrix

    train_inputs = make_rel_ext_inputs(train_words, embed, train_pos, pos_to_index, 
                                       train_ent_labels, label_to_index,
                                       train_dep, dep_to_index,
                                       train_branch1, train_branch2)
    train_outputs = make_ner_outputs(train_rel_labels)

    test_inputs = make_rel_ext_inputs(test_words, embed, test_pos, pos_to_index,
                                      test_ent_labels, label_to_index,
                                      test_dep, dep_to_index,
                                      test_branch1, test_branch2)

    model = build_rel_ext_model(len(label_classes), embed_matrix, len(pos_classes))

    train_generator = DataGenerator(train_inputs, train_outputs, 32)

    evaluator = ModelEval(DataGenerator(test_inputs),
                          test_ent_labels,
                          index_to_label)

    model_saver = ModelCheckpoint(filepath='./checkpoints/' + model.name.replace(' ', '_') + '_{epoch:02d}.hdf5',
                                  verbose=1, save_best_only=True, monitor='valid_f1')

    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    csv_logger = CSVLogger(f"./logs/RE_log_{time_stamp}.csv", append=False)

    # model.load_weights('./checkpoints/char_cnn_bilstm_crf17.hdf5')

    model.fit_generator(train_generator, epochs=1, callbacks=[evaluator, model_saver, csv_logger])

    test_pred_indices = predict(model, DataGenerator(test_inputs))

    test_pred_labels = transform_indices_to_labels(index_to_label, test_pred_indices)

    print(compute_f1(test_pred_labels, test_ent_labels))

    #with open('ner_results.txt', 'wt') as file:
    #    for sent, sent_true_labels, sent_pred_labels in zip(test_data, test_labels, test_pred_labels):
    #        file.writelines([' '.join(z) + '\n' for z in zip(sent, sent_true_labels, sent_pred_labels)])
    #        file.write('\n')


if __name__ == '__main__':
    main()
