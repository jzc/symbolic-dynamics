import networkx as nx

from symbolic_dynamics.sofic import from_partial_fns


def gm_shift():
    G = nx.MultiDiGraph()
    G.add_edge(0, 0, label="0")
    G.add_edge(0, 1, label="1")
    G.add_edge(1, 0, label="0")
    return G


def m_shift(m):
    G = nx.MultiDiGraph()
    G.add_edge(0, 0, label="1")
    nx.add_cycle(G, range(m), label="0")
    return G


def even_shift():
    return m_shift(2)


def cernys_automata(n):
    d = {
        "a": {0: 1} | {i: i for i in range(1, n)},
        "b": {i: (i+1) % n for i in range(n)},
    }
    return from_partial_fns(d)
