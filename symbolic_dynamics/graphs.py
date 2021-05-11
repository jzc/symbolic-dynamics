"""Functions for generating specific instances of graphs
related to symbolic dynamics."""

import networkx as nx

from symbolic_dynamics.sofic import from_partial_fns


def golden_mean_shift():
    """Returns the minimal determinstic presentation of the
    golden mean shift.

    The :dfn:`golden mean shift` is the subshift of the
    full {0,1}-shift such that word 11 occurs in no point
    of the subshift.

    Examples
    --------
    >>> G = sd.golden_mean_shift()
    >>> len(sd.idot(G, G, "11"))
    0

    """
    G = nx.MultiDiGraph()
    G.add_edge(0, 0, label="0")
    G.add_edge(0, 1, label="1")
    G.add_edge(1, 0, label="0")
    return G


def mn_gap_shift(m):
    """Returns the minimal deterministic presentation of
    the mn-gap shift.

    The :dfn:`mn-gap shift` is the subshift of the full {0,1}-shift
    such that between any two 1's within a point
    of the subshift, the number of 0's occurring between them
    is a multiple of m.

    Parameters
    ----------
    m : int

    Examples
    --------
    >>> G = mn_gap_shift(3)
    >>> len(sd.idot(G, G, "1001"))
    0

    """
    G = nx.MultiDiGraph()
    G.add_edge(0, 0, label="1")
    nx.add_cycle(G, range(m), label="0")
    return G


def even_shift():
    """Returns the minimal deterministic presentation of
    the even shift.

    The :dfn:`even shift` is the subshift of the full {0,1}-shift
    such that between any two 1's within a point
    of the subshift, there are an even number of 0's occuring
    between them. Equivalently, the even shift is the 2n-gap shift.

    Examples
    --------
    >>> G = even_shift()
    >>> len(sd.idot(G, G, "101"))
    0

    """
    return mn_gap_shift(2)


def cernys_automata(n):
    """Returns the `n`\ th Cerny automata.

    The :dfn:`nth Cerny automata` is a fully deterministic graph `C_n`
    with states over {0, 1, ..., n-1} with alphabet
    {a, b}  with the following transition action:

    * ``sd.dot(C_n, i, "b") == i`` for ``i in range(n)``
    * ``sd.dot(C_n, 0, "a") == 1``
    * ``sd.dot(C_n, i, "a") == i`` for ``i in range(1, n)``

    The shortest synchronizing word for `C_n` is the word
    ``("a"+"b"*(n-1))*(n-2)+"a"``, which has length of ``(n-1)**2``.

    Parameters
    ----------
    n : int

    Examples
    --------
    >>> n = 5
    >>> C_n = cernys_automata(n)
    >>> w = ("a"+"b"*(n-1))*(n-2)+"a"
    >>> len(sd.idot(C_n, C_n, w))
    1

    """
    d = {
        "a": {**{0: 1}, **{i: i for i in range(1, n)}},
        "b": {i: (i+1) % n for i in range(n)},
    }
    return from_partial_fns(d)
