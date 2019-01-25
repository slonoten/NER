import numpy as np

from embeddings import load_embeddings


def test_fast_text_fixed_size():
    vocab, matrix = load_embeddings('./data/fast_text.txt', 4, is_fast_text=True)
    assert (matrix[vocab['a']] == np.ones((1,))).all()
    assert (matrix[vocab['c']] == np.ones((1,)) * 3).all()


def test_fast_text_set():
    vocab, matrix = load_embeddings('./data/fast_text.txt', word_set={'a', 'b', 'c'}, is_fast_text=True)
    assert matrix.shape == (5, 3)
    assert len(vocab) == 3
    assert (matrix[vocab['a']] == np.ones((1,))).all()
    assert (matrix[vocab['c']] == np.ones((1,)) * 3).all()


def test_word2vec_size():
    vocab, matrix = load_embeddings('./data/word2vec.txt', 4)
    assert (matrix[vocab['a']] == np.ones((1,))).all()
    assert (matrix[vocab['c']] == np.ones((1,)) * 3).all()


def test_word2vec_set():
    vocab, matrix = load_embeddings('./data/word2vec.txt', word_set={'a', 'b', 'c'})
    assert matrix.shape == (5, 3)
    assert len(vocab) == 3
    assert (matrix[vocab['a']] == np.ones((1,))).all()
    assert (matrix[vocab['c']] == np.ones((1,)) * 3).all()