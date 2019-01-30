"""Parsing CONLL format"""


from typing import List, Tuple, Iterable


def load_conll(file_name: str, indices: List[int],
               ignore_tokens: Iterable[str] = None, max_sentences: int = 0) -> Tuple:
    ignore_set = set(ignore_tokens) if ignore_tokens else {}
    new_sentence = False
    with open(file_name) as file:
        data = [[[]] for _ in indices]
        min_row_len = max(indices) + 1
        for s in file:
            items = s.strip().split()
            if len(items) >= min_row_len:
                if new_sentence:
                    if max_sentences and len(data[0]) > max_sentences:
                        break
                    if len(data[0][-1]) > 0:
                        for d in data:
                            d.append([])
                    new_sentence = False
                if items[indices[0]] in ignore_set:
                    continue
                for i, d in zip(indices, data):
                    d[-1].append(items[i])
            else:
                new_sentence = True

    return data
