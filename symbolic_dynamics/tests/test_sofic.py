import networkx as nx

from symbolic_dynamics import sofic as s
from symbolic_dynamics.sofic import iset

def path(n, label):
    G = nx.MultiDiGraph()
    nx.add_path(G, range(n), label=label)
    return G


def test_dot():
    for label in ["a", "aa"]:
        G = path(5, label)
        assert s.dot(G, 0, [label]*4) == 4
        assert s.dot(G, 0, [label]*5) is None
        

def test_idot():
    for label in ["a", "aa"]:
        G = path(5, label)
        for i in range(6):
            assert len(s.idot(G, G, [label]*i)) == (5-i)


def test_alphabet():
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, label="a")
    G.add_edge(2, 1, label="b")
    G.add_edge(3, 3, label="ab")
    G.add_edge(3, 3, other="c")
    sigma = s.alphabet(G)
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
    ol = s.out_labels(G, 1)
    assert "a" in ol
    assert "b" not in ol
    assert "c" in ol
    assert "d" not in ol
    assert "e" not in ol
    

def test_essential():
    G = path(5, "a")
    assert not s.is_essential(G)
    s.make_essential(G)
    assert len(G) == 0
    assert s.is_essential(G)
    
    G = path(5, "a")
    G.add_edge(0, 0)
    assert not s.is_essential(G)
    s.make_essential(G)
    assert list(G) == [0]
    assert s.is_essential(G)

    G = path(5, "a")
    G.add_edge(4, 4)
    assert not s.is_essential(G)
    s.make_essential(G)
    assert list(G) == [4]
    assert s.is_essential(G)

    G = path(5, "a")
    G.add_edge(0, 0)
    G.add_edge(4, 4)
    assert s.is_essential(G)
    s.make_essential(G)
    assert len(G) == 5
    assert s.is_essential(G)
    

def test_deterministic():
    G = path(5, "a")
    assert s.is_deterministic(G)
    assert not s.is_fully_deterministic(G)

    G.add_edge(4, 4, label="a")
    assert s.is_deterministic(G)
    assert s.is_fully_deterministic(G)

    G.add_edge(0, 0, label="b")
    assert s.is_deterministic(G)
    assert not s.is_fully_deterministic(G)

    G.add_edge(0, 0, label="a")
    assert not s.is_deterministic(G)
    assert not s.is_fully_deterministic(G)


def test_from_partial_fns():
    d = {"a": {i: i+1 for i in range(4)}, "b": {4: 4}}
    G = s.from_partial_fns(d)
    for i in range(4):
        assert s.dot(G, i, "a") == i+1
        assert s.dot(G, i, "b") is None
    assert s.dot(G, 4, "a") is None
    assert s.dot(G, 4, "b") == 4


def test_is_irreducible():
    G = path(5, "a")
    assert not s.is_irreducible(G)
    G.add_edge(4, 0)
    assert s.is_irreducible(G)


def test_partition():
    partition = s.Partition(range(6))
    
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

    partition.split([2,3])
    assert invariant()
    assert partition.part_count() == 5

    partition.split([0,1,2,3])
    assert invariant()
    assert partition.part_count() == 5

    assert (set(iset(part) for part in partition.parts.values())
            == set([iset([0]), iset([1]), iset([2]), iset([3]), iset([4,5])]))
    

def test_hopcroft():
    G = nx.MultiDiGraph()
    nx.add_path(G, range(0, 5), label="a")
    G.add_edge(4, 4, label="a")
    nx.add_path(G, range(10, 15), label="a")
    G.add_edge(14, 14, label="a")
    for i in range(0, 5):
        G.add_edge(i, i, label="b")
        G.add_edge(i+10, i+10, label="b")
    assert s.is_fully_deterministic(G)
    
    partiton = s.hopcroft(G, [4, 14])
    for i in range(0, 5):
        assert partiton.part_lookup[i] == partiton.part_lookup[i+10]

    G.remove_node(10)
    assert s.is_fully_deterministic(G)

    partiton = s.hopcroft(G, [4, 14])
    for i in range(1, 5):
        assert partiton.part_lookup[i] == partiton.part_lookup[i+10]

    assert len(partiton.parts[partiton.part_lookup[0]])

    G = nx.MultiDiGraph()
    nx.add_cycle(G, range(5), label="a")
    nx.add_cycle(G, reversed(range(5)), label="b")
    partiton = s.hopcroft(G, [0])
    assert partiton.part_count() == 5


def test_add_sink():
    G = path(5, "a")
    s.add_sink_vertex(G)
    assert s.dot(G, 4, "a") == s.sink
    for i in range(4):
        assert s.dot(G, i, "a") != s.sink

    G = path(5, "a")
    s.add_sink_vertex(G, sigma=["a", "b"])
    assert s.dot(G, 4, "a") == s.sink
    for i in range(4):
        assert s.dot(G, i, "a") != s.sink
    for i in range(5):
        assert s.dot(G, i, "b") == s.sink
    
    G = path(5, "a")
    G.add_edge(4, 4, label="b")
    # assert 
