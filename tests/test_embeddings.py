import numpy as np

from embeddings import Embeddings


def test_fast_text_fixed_size():
    embed = Embeddings('./data/fast_text.txt', True, 4, is_fast_text=True)
    matrix = embed.matrix
    assert (matrix[embed['a']] == np.ones((1,))).all()
    assert (matrix[embed['c']] == np.ones((1,)) * 3).all()


def test_fast_text_set():
    embed = Embeddings('./data/fast_text.txt', True, word_set={'a', 'b', 'c'}, is_fast_text=True)
    matrix = embed.matrix
    assert matrix.shape == (5, 3)
    assert len(embed.vocab) == 3
    assert (matrix[embed['a']] == np.ones((1,))).all()
    assert (matrix[embed['c']] == np.ones((1,)) * 3).all()


def test_word2vec_size():
    embed = Embeddings('./data/word2vec.txt', True, 4)
    matrix = embed.matrix
    assert (matrix[embed['a']] == np.ones((1,))).all()
    assert (matrix[embed['c']] == np.ones((1,)) * 3).all()


def test_word2vec_set():
    embed = Embeddings('./data/word2vec.txt', True, word_set={'a', 'b', 'c'})
    matrix = embed.matrix
    assert matrix.shape == (5, 3)
    assert len(embed.vocab) == 3
    assert (matrix[embed['a']] == np.ones((1,))).all()
    assert (matrix[embed['c']] == np.ones((1,)) * 3).all()


def test_word2vec_case_sensitive():
    embed = Embeddings('./data/w2v_case_sensitive.txt', False, 4)
    matrix = embed.matrix
    assert (matrix[embed['A']] == np.ones((1,))).all()
    assert (matrix[embed['a']] == np.ones((1,)) * 2).all()