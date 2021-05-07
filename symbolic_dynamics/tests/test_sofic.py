import networkx as nx

import symbolic_dynamics as sd
from symbolic_dynamics.utils import iset


def path(n, label):
    G = nx.MultiDiGraph()
    nx.add_path(G, range(n), label=label)
    return G


def test_dot():
    for label in ["a", "aa"]:
        G = path(5, label)
        assert sd.dot(G, 0, [label]*4) == 4
        assert sd.dot(G, 0, [label]*5) is None


def test_idot():
    for label in ["a", "aa"]:
        G = path(5, label)
        for i in range(6):
            assert len(sd.idot(G, G, [label]*i)) == (5-i)


def test_alphabet():
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, label="a")
    G.add_edge(2, 1, label="b")
    G.add_edge(3, 3, label="ab")
    G.add_edge(3, 3, other="c")
    sigma = sd.alphabet(G)
    assert "a" in sigma
    assert "b" in sigma
    assert "ab" in sigma
    assert "c" not in sigma


def test_out_labels():
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, label="a")
    G.add_edge(2, 1, label="b")
    G.add_edge(1, 1, label="c")
    G.add_edge(1, 1, other="d")
    G.add_edge(2, 1, other="e")
    ol = sd.out_labels(G, 1)
    assert "a" in ol
    assert "b" not in ol
    assert "c" in ol
    assert "d" not in ol
    assert "e" not in ol


def test_essential():
    G = path(5, "a")
    assert not sd.is_essential(G)
    sd.make_essential(G)
    assert len(G) == 0
    assert sd.is_essential(G)

    G = path(5, "a")
    G.add_edge(0, 0)
    assert not sd.is_essential(G)
    sd.make_essential(G)
    assert list(G) == [0]
    assert sd.is_essential(G)

    G = path(5, "a")
    G.add_edge(4, 4)
    assert not sd.is_essential(G)
    sd.make_essential(G)
    assert list(G) == [4]
    assert sd.is_essential(G)

    G = path(5, "a")
    G.add_edge(0, 0)
    G.add_edge(4, 4)
    assert sd.is_essential(G)
    sd.make_essential(G)
    assert len(G) == 5
    assert sd.is_essential(G)


def test_deterministic():
    G = path(5, "a")
    assert sd.is_deterministic(G)
    assert not sd.is_fully_deterministic(G)

    G.add_edge(4, 4, label="a")
    assert sd.is_deterministic(G)
    assert sd.is_fully_deterministic(G)

    G.add_edge(0, 0, label="b")
    assert sd.is_deterministic(G)
    assert not sd.is_fully_deterministic(G)

    G.add_edge(0, 0, label="a")
    assert not sd.is_deterministic(G)
    assert not sd.is_fully_deterministic(G)

    G = nx.MultiDiGraph()
    G.add_edge(1, 2, label="a")
    G.add_edge(1, 2, label="b")
    G.add_edge(2, 1, label="b")
    G.add_edge(2, 1, label="c")
    assert sd.is_deterministic(G)
    assert not sd.is_fully_deterministic(G)


def test_from_partial_fns():
    d = {"a": {i: i+1 for i in range(4)}, "b": {4: 4}}
    G = sd.from_partial_fns(d)
    for i in range(4):
        assert sd.dot(G, i, "a") == i+1
        assert sd.dot(G, i, "b") is None
    assert sd.dot(G, 4, "a") is None
    assert sd.dot(G, 4, "b") == 4


def test_is_irreducible():
    G = path(5, "a")
    assert not sd.is_irreducible(G)
    G.add_edge(4, 0)
    assert sd.is_irreducible(G)


def test_partition():
    partition = sd.Partition(range(6))

    def invariant():
        return all(e in partition.parts[partition.part_lookup[e]]
                   for e in partition.elements)

    assert invariant()
    assert partition.part_count() == 1

    partition.split([0])
    assert invariant()
    assert partition.part_count() == 2

    partition.split([0])
    assert invariant()
    assert partition.part_count() == 2

    partition.split([1, 2])

    assert invariant()
    assert partition.part_count() == 3

    partition.split([2, 3])
    assert invariant()
    assert partition.part_count() == 5

    partition.split([0, 1, 2, 3])
    assert invariant()
    assert partition.part_count() == 5

    assert (set(iset(part) for part in partition.parts.values())
            == set([iset([0]), iset([1]), iset([2]), iset([3]), iset([4, 5])]))


def test_hopcroft():
    G = nx.MultiDiGraph()
    nx.add_path(G, range(0, 5), label="a")
    G.add_edge(4, 4, label="a")
    nx.add_path(G, range(10, 15), label="a")
    G.add_edge(14, 14, label="a")
    for i in range(0, 5):
        G.add_edge(i, i, label="b")
        G.add_edge(i+10, i+10, label="b")
    assert sd.is_fully_deterministic(G)

    partiton = sd.hopcroft(G, [4, 14])
    for i in range(0, 5):
        assert partiton.part_lookup[i] == partiton.part_lookup[i+10]

    G.remove_node(10)
    assert sd.is_fully_deterministic(G)

    partiton = sd.hopcroft(G, [4, 14])
    for i in range(1, 5):
        assert partiton.part_lookup[i] == partiton.part_lookup[i+10]

    assert len(partiton.parts[partiton.part_lookup[0]])

    G = nx.MultiDiGraph()
    nx.add_cycle(G, range(5), label="a")
    nx.add_cycle(G, reversed(range(5)), label="b")
    partiton = sd.hopcroft(G, [0])
    assert partiton.part_count() == 5


def test_add_sink():
    G = path(5, "a")
    sd.add_sink_vertex(G)
    assert sd.dot(G, 4, "a") == sd.sink
    for i in range(4):
        assert sd.dot(G, i, "a") is not None
        assert sd.dot(G, i, "a") != sd.sink

    G = path(5, "a")
    sd.add_sink_vertex(G, sigma=["a", "b"])
    assert sd.dot(G, 4, "a") == sd.sink
    for i in range(4):
        assert sd.dot(G, i, "a") is not None
        assert sd.dot(G, i, "a") != sd.sink
    for i in range(5):
        assert sd.dot(G, i, "b") == sd.sink

    G = path(5, "a")
    G.add_edge(4, 4, label="b")
    sd.add_sink_vertex(G)
    assert sd.dot(G, 4, "b") is not None
    assert sd.dot(G, 4, "b") != sd.sink
    assert sd.dot(G, 4, "a") == sd.sink
    for i in range(4):
        assert sd.dot(G, i, "a") is not None
        assert sd.dot(G, i, "a") != sd.sink
        assert sd.dot(G, i, "b") == sd.sink


def test_get_follower_equivalences():
    G = nx.MultiDiGraph()
    G.add_edge(1, 3, label="a")
    G.add_edge(2, 3, label="a")
    for i in range(3):
        G.add_edge(i, i, label="b")
    parts, part_lookup = sd.get_follower_equivalences(G)
    assert part_lookup[1] == part_lookup[2]

    G = path(5, "a")
    nx.add_path(G, range(10, 15), label="a")
    G.add_edge(0, 0, label="b")
    G.add_edge(4, 4, label="b")
    G.add_edge(10, 10, label="b")
    G.add_edge(14, 14, label="b")
    parts, part_lookup = sd.get_follower_equivalences(G)
    for i in range(5):
        assert part_lookup[i] == part_lookup[i+10]


def test_is_follower_separated():
    G = nx.MultiDiGraph()
    G.add_edge(1, 1, label="a")
    G.add_edge(1, 2, label="b")
    G.add_edge(2, 1, label="b")
    assert sd.is_follower_separated(G)


def test_find_synchronizing_word():
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, label="b")
    G.add_edge(2, 1, label="b")
    assert sd.find_synchronizing_word(G) is None

    G.add_edge(1, 1, label="a")
    w = sd.find_synchronizing_word(G)
    assert w is not None
    assert len(sd.idot(G, G, w)) == 1


def test_reduce():
    G = path(5, "a")
    nx.add_path(G, range(10, 15), label="a")
    G.add_edge(0, 0, label="b")
    G.add_edge(4, 4, label="b")
    G.add_edge(10, 10, label="b")
    G.add_edge(14, 14, label="b")
    Gr = sd.reduce(G)
    assert len(Gr) == 5
    assert sd.is_follower_separated(Gr)


def test_find_separating_word():
    G = nx.MultiDiGraph()
    nx.add_cycle(G, range(2), label="b")
    G.add_edge(0, 0, label="a")

    H = nx.MultiDiGraph()
    nx.add_cycle(H, range(3), label="b")
    H.add_edge(0, 0, label="a")

    w = sd.find_separating_word(G, H)
    assert w is not None
    assert len(sd.idot(G, G, w)) > 0
    assert len(sd.idot(H, H, w)) == 0

    H = nx.MultiDiGraph()
    nx.add_cycle(H, range(4), label="b")
    H.add_edge(0, 0, label="a")
    assert sd.is_subshift(H, G)


def test_is_synchronizing():
    G = nx.MultiDiGraph()
    G.add_edge(1, 1, label="a")
    G.add_edge(1, 2, label="b")
    G.add_edge(2, 2, label="a")
    assert not sd.is_synchronizing(G)

    G = nx.MultiDiGraph()
    G.add_edge(1, 1, label="c")
    G.add_edge(1, 2, label="b")
    G.add_edge(2, 2, label="a")
    assert sd.is_synchronizing(G)


def test_is_label_isomorphic():
    G = nx.MultiDiGraph()
    nx.add_cycle(G, range(5), label="b")
    G.add_edge(0, 0, label="a")
    H = nx.relabel_nodes(G, lambda x: 5-x)
    assert sd.is_follower_separated(G)
    assert sd.is_follower_separated(H)
    assert sd.is_label_isomorphic_fs(G, H)

    H.add_edge(-1, 0, label="a")
    H.add_edge(-1, -1, label="b")
    assert sd.is_follower_separated(H)
    assert not sd.is_label_isomorphic_fs(G, H)


def test_subset_construction():
    G = nx.MultiDiGraph()
    G.add_edge(1, 1, label="1")
    G.add_edge(1, 1, label="0")
    G.add_edge(1, 2, label="1")
    G.add_edge(2, 3, label="1")
    ssc = sd.subset_presentation(G)
    assert sd.is_deterministic(ssc)
    assert 3 in sd.dot(ssc, iset(G), "011")
    assert 3 not in sd.dot(ssc, iset(G), "01")
    assert 3 not in sd.dot(ssc, iset(G), "0110")
