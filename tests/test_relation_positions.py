from rel_ext import get_entity_positions, get_range_index, get_path_to_root, get_tree_path


def test_get_entity_positions():
    e1, e2 = get_entity_positions('8001	"The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."')
    assert e1 == (16, 22), "Positions not matched"
    assert e2 == (34, 39), "Positions not matched"


def test_get_entity_positions_2():
    e1, e2 = get_entity_positions('213	"In a former residence, your writer had an atrium containing a <e1>duel</e1> of doves<e2>moles</e2> numbering 22."')
    assert e1[0] == 62, "Positions not matched"
    assert e2[0] == 75, "Positions not matched"



def test_get_range_indices():
    indices = get_range_index((16, 22),
                              [(0, 3), (4, 8), (9, 15), (16, 22), (23, 27),
                               (28, 33), (34, 39), (40, 43), (44, 53)])
    assert indices == 3


def test_get_path_to_root():
    #          1  2  3  4  5  6  7  8  9  10
    depends = [4, 3, 4, 5, 0, 5, 6, 7, 7, -1]
    path = list(get_path_to_root(depends, 0))
    assert [0, 3, 4] == path
    path = list(get_path_to_root(depends, 8))
    assert [8, 6, 5, 4] == path
    path = list(get_path_to_root(depends, 1))
    assert [1, 2, 3, 4] == path


def test_get_tree_path():
    #          1  2  3  4  5  6  7  8  9  10
    depends = [4, 3, 4, 5, 0, 5, 6, 7, 7, -1]
    path1, path2 = list(get_tree_path(depends, 0, 1))
    assert [0, 3] == path1
    assert [1, 2, 3] == path2
    path1, path2 = list(get_tree_path(depends, 0, 8))
    assert [0, 3, 4] == path1
    assert [8, 6, 5, 4] == path2
    path1, path2 = list(get_tree_path(depends, 0, 0))
    assert path1 == [0]
    assert path2 == [0]

