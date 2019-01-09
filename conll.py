"""Parsing CONLL format"""


from typing import List, Tuple


def load_conll(file_name: str, column_index: int = 3) -> Tuple[List[List[str]], List[List[str]]]:
    with open(file_name) as file:
        sentences = [[]]
        labels = [[]]
        for s in file:
            s = s.strip()
            ss = s.split()
            if len(ss) == 4 and ss[0] != '-DOCSTART-':
                sentences[-1].append(ss[0])
                labels[-1].append(ss[column_index])
            else:
                if len(sentences[-1]) > 0:
                    sentences.append([])
                    labels.append([])
        if sentences[-1]:
            return sentences, labels
        else:
            return sentences[:-1], labels[:-1]

