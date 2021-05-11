"""Functions and algorithms for sofic shifts.

The functions in this module are designed to work
on a :class:`networkx.MultiDiGraph`, which we refer to just
as :dfn:`graphs`. Furthermore, some functions require
more specific properties to ensure correctness, which include

* labeled - :func:`is_labeled`
* deterministic - :func:`is_deterministic`
* essential - :func:`is_essential`
* irreducible - :func:`is_irreducible`
* follower-separated - :func:`is_follower_separated`
* synchronizing - :func:`is_synchronizing`

See the documentation for the associated functions for a definition
of the properties. However, all functions in this module
do not explicity check if their input graphs have the property needed
correctness.
"""


import networkx as nx
from symbolic_dynamics.utils import iset, first
from itertools import combinations, islice, product
import random
from contextlib import contextmanager
from functools import wraps

# __all__ = ["dot", "idot", "is_labeled", "alphabet", "out_labels",
#            "is_stranded", "is_essential", "make_essential",
#            "is_deterministic", "is_fully_deterministic",
#            "from_partial_fns", "random_deterministic_graph",
#            "random_deterministic_graph_with_props",
#            "is_irreducible", "add_sink_vertex",
#            "get_follower_equivalences", "is_follower_separated",
#            "extend_to_synchronizing_word", "find_synchronizing_word",
#            "reduce", "find_separating_word", "is_subshift",
#            "is_synchronizing", "is_label_isomorphic_fs",
#            "subset_construction", "sink"]


def dot(G, q, w):
    """Computes the transition action of `w` in `G` on `q`.

    In a deterministic labeled graph `G`, any path starting at a
    given vertex is uniquely determinied by its sequence of labels.
    If there is a path labeled `w` starting at `q` in `G`, then
    the transition action of `w` in `G` on `q` is defined to be
    the vertex that path ends at. Otherwise, if there is no such
    path, then the transtion action of `w` in `G` on `q` is
    defined to be None.

    Parameters
    ----------
    G : deterministic labeled graph
    q : vertex in `G`
    w : word

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> nx.add_path(G, range(5), label="a")
    >>> sd.dot(G, 0, "aaaa")
    4
    >>> sd.dot(G, 1, "aaaa") is None
    True

    """
    if q is None:
        return None

    for a in w:
        delta = {ell: p for (_, p, ell) in G.out_edges(q, data="label")}
        if a not in delta:
            return None
        q = delta[a]

    return q


def idot(G, it, w):
    """Computes the transition action of `w` in `G` on each element
    of the iterable `it`, returning an :class:`.iset`
    of the non-None results.

    Parameters
    ----------
    G : deterministic labeled graph
    it : iterable of vertices in `G`
    w : word

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> nx.add_path(G, range(5), label="a")
    >>> sd.idot(G, [0, 1], "aaa")
    {3, 4}
    >>> sd.idot(G, [0, 1], "aaaa")
    {4}
    >>> sd.idot(G, [0, 1], "aaaaa")
    {}

    See Also
    --------
    :func:`dot`

    """
    x = (dot(G, q, w) for q in it)
    x = (q for q in x if q is not None)
    return iset(x)


def is_labeled(G):
    """Returns True iff every edge in the graph `G`
    has the attribute ``label``.

    Parameters
    ----------
    G : graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 2, label="a")
    >>> sd.is_labeled(G)
    True
    >>> G.add_edge(2, 1)
    >>> sd.is_labeled(G)
    False

    """
    return all("label" in data for _, _, data in G.edges(data=True))


def alphabet(G):
    """Returns an :class:`.iset` of labels appearing in `G`.

    Parameters
    ----------
    G : labeled graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 2, label="a")
    >>> G.add_edge(2, 1, label="b")
    >>> sd.alphabet(G)
    {'a', 'b'}

    """
    return iset(label for (_, _, label) in G.edges(data="label"))


def out_labels(G, q):
    """Returns a list of each of the labels appearing on the edges
    starting at `q` in `G`.

    Parameters
    ----------
    G : labeled graph
    q : vertex in `G`

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 2, label="a")
    >>> G.add_edge(1, 3, label="a")
    >>> G.add_edge(1, 1, label="b")
    >>> sd.out_labels(G, 1)
    ['a', 'a', 'b']

    """
    return [label for (_, _, label) in G.out_edges(q, data="label")]


def is_stranded(G, q):
    """Returns True iff `q` is stranded in `G`.

    A vertex `q` in `G` is :dfn:`stranded` if there is no
    edge starting at `q` or if there is no edge ending at `q`.

    Parameters
    ----------
    G : graph
    q : vertex in `G`

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 2)
    >>> sd.is_stranded(G, 1)
    True
    >>> G.add_edge(1, 1)
    >>> sd.is_stranded(G, 1)
    False

    See Also
    --------
    :func:`is_essential`

    """
    return not (G.out_edges(q) and G.in_edges(q))


def is_essential(G):
    """Returns True iff `G` is essential.

    A graph `G` is :dfn:`essential` if no vertex is stranded.

    Parameters
    ----------
    G : graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 2)
    >>> G.add_edge(1, 1)
    >>> sd.is_essential(G)
    False
    >>> G.add_edge(2, 2)
    >>> sd.is_essential(G)
    True

    See Also
    --------
    :func:`is_stranded`

    """
    return all(not is_stranded(G, q) for q in G)


def make_essential(G):
    """Modifies `G` by removing vertices in `G` that do not lie
    on bi-infinite paths. The resulting graph is the maximal essential
    subgraph of the original graph.

    Parameters
    ----------
    G : a graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 2)
    >>> G.add_edge(1, 1)
    >>> len(G), sd.is_essential(G)
    (2, False)
    >>> sd.make_essential(G)
    >>> len(G), sd.is_essential(G)
    (1, True)

    See Also
    --------
    :func:`is_essential`

    """
    nonextensible = [q for q in G if not G.out_edges(q)]
    while True:
        frontier = iset(q for (q, _) in G.in_edges(nonextensible))
        G.remove_nodes_from(nonextensible)
        nonextensible = [q for q in frontier if not G.out_edges(q)]
        if not nonextensible:
            break

    noncoextensible = [q for q in G if not G.in_edges(q)]
    while True:
        frontier = iset(q for (_, q) in G.out_edges(noncoextensible))
        G.remove_nodes_from(noncoextensible)
        noncoextensible = [q for q in frontier if not G.in_edges(q)]
        if not noncoextensible:
            break


def is_deterministic(G):
    """Returns True iff `G` is deterministic.

    A labeled graph `G` is :dfn:`deterministic` if every
    edge starting at a given vertex is labeled uniquely.

    Parameters
    ----------
    G : labeled graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 2, label="a")
    >>> G.add_edge(1, 3, label="b")
    >>> sd.is_deterministic(G)
    True
    >>> G.add_edge(1, 4, label="a")
    >>> sd.is_deterministic(G)
    False

    """
    for q in G:
        ol = out_labels(G, q)
        if len(ol) != len(set(ol)):
            return False
    return True


def is_fully_deterministic(G):
    """Returns True iff `G` is fully deterministic.

    A labeled graph `G` is :dfn:`fully deterministic` if
    `G` is deterministic and for every vertex `q` in `G`,
    the set of labels of edges starting at `q` is equal
    the set of labels appearing in `G`.

    Parameters
    ----------
    G : labeled graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 2, label="a")
    >>> G.add_edge(2, 1, label="a")
    >>> sd.is_fully_deterministic(G)
    True
    >>> G.add_edge(1, 1, label="b")
    >>> sd.is_fully_deterministic(G)
    False
    >>> G.add_edge(2, 2, label="b")
    >>> sd.is_fully_deterministic(G)
    True

    See Also
    --------
    :func:`is_deterministic`, :func:`alphabet`

    """
    sigma = alphabet(G)
    for q in G:
        ol = out_labels(G, q)
        ol_s = iset(ol)
        if not (len(ol) == len(ol_s) and ol_s == sigma):
            return False
    return True


def from_partial_fns(pfns):
    """Returns a deterministic graph built from given partial functions.

    Here, `pfns` is a dict of dicts: the keys of the outer dicts are labels
    to be used in the output graph, and the inner dicts represent where the
    edges for a given label are located.

    More specifically, if `G` is the result of ``from_partial_fns(pfns)``,
    then `G` has the following property:

    >>> for (a, pfn) in pfns.items():
    ...     for q in G:
    ...         if q in pfn:
    ...              assert dot(G, q, [a]) == pfn[q]
    ...         else:
    ...              assert dot(G, q, [a]) is None

    Parameters
    ----------
    pfns : dict of dicts; see above

    Examples
    --------
    >>> d = {"a": {0: 1, 1: 2, 2: 3}, "b": {3: 0, 0: 3}}
    >>> G = sd.from_partial_fns(d)
    >>> list(G.edges(data="label"))
    [(0, 1, 'a'), (0, 3, 'b'), (1, 2, 'a'), (2, 3, 'a'), (3, 0, 'b')]

    """
    G = nx.MultiDiGraph()
    for a, pfn in pfns.items():
        for p, q in pfn.items():
            G.add_edge(p, q, label=a)
    return G


def random_partial_fn(n):
    pfn = {}
    for i in range(n):
        r = random.randrange(-1, n)
        if r >= 0:
            pfn[i] = r
    return pfn


def random_deterministic_graph(n, m):
    """Randomly generates a deterministic graph defined by
    `m` partial functions over `n` states.

    Parameters
    ----------
    n : int
     domain/codomain of partial functions
    m : int
     number of partial functions
    """
    return from_partial_fns({str(i): random_partial_fn(n) for i in range(m)})


def random_deterministic_graph_with_props(n, m, props):
    """Randomly generates a deterministic graph defined by
    `m` partial functions over `n` states that satisfies each
    each predicate in `props`.

    Parameters
    ----------
    n : int
     domain/codomain of partial functions
    m : int
     number of partial functions
    props : list of predicates
     the predicates for the random graph to satisfy

    Examples
    --------
    >>> is_reducible = lambda G: not sd.is_irreducible(G)
    >>> props = [sd.is_essential, is_reducible]
    >>> G = sd.random_deterministic_graph_with_props(5, 2, props)
    >>> sd.is_essential(G)
    True
    >>> sd.is_irreducible(G)
    False

    """
    while True:
        G = random_deterministic_graph(n, m)
        if all(p(G) for p in props):
            return G


def subset_presentation(G):
    """Constructs the subset presentation of `G`.

    The :dfn:`subset presentation` of `G` is a deterministic
    labeled graph such that shift presented by `G` is equivalent
    to the shift presented by the subset presentation.

    Parameters
    ----------
    G : labeled graph
    """
    ssc = nx.MultiDiGraph()
    sigma = alphabet(G)
    X = iset(G)
    stack = [X]
    popped = set()
    while stack:
        X = stack.pop()
        ssc.add_node(X)
        popped.add(X)
        for a in sigma:
            Y = iset(q for (_, q, b) in G.out_edges(X, data="label") if b == a)
            if Y:
                ssc.add_edge(X, Y, label=a)
                if Y not in popped:
                    stack.append(Y)
    return ssc


def is_irreducible(G):
    """Returns True iff `G` is irreducible

    A graph is :dfn:`irreducible` if for any two vertices in `G`,
    there is a path between them. Synonymous with *strongly connected*.

    Parameters
    ----------
    G : graph

    """
    return nx.is_strongly_connected(G)


class Partition:
    def __init__(self, elements):
        self.elements = elements
        self.part_lookup = {e: 0 for e in elements}
        self.parts = {0: set(elements)}

    def part_count(self):
        return len(self.parts)

    def part_size(self, p):
        return len(self.parts[p])

    def split(self, mark_it):
        # mark_it is an iterable of elements to mark
        marks = {}
        for element in mark_it:
            p = self.part_lookup[element]
            if p not in marks:
                marks[p] = set()
            marks[p].add(element)

        splits = []
        # iterate over parts with marks
        for p in marks:
            if self.part_size(p) > len(marks[p]):
                new_p = self.part_count()
                self.parts[new_p] = set()
                for marked in marks[p]:
                    self.part_lookup[marked] = new_p
                    self.parts[new_p].add(marked)
                    self.parts[p].remove(marked)
                splits.append((p, new_p))

        return splits

    def select_smaller(self, p1, p2):
        if self.part_size(p1) <= self.part_size(p2):
            return p1
        else:
            return p2


def hopcroft(G, S):
    """Hopcroft's algorthm for computing state equivalence.

    Parameters
    ----------
    G : fully deterministic graph
    S : iterable
        one half of the initial (bi)partition

    Returns
    -------
    Partition
    """
    sigma = alphabet(G)
    partition = Partition(list(G))
    p1, p2 = partition.split(S)[0]

    smaller = partition.select_smaller(p1, p2)
    wait_set = set()
    for a in sigma:
        wait_set.add((smaller, a))

    while wait_set:
        p, a = wait_set.pop()
        inv_a_p = G.in_edges(partition.parts[p], data="label")
        inv_a_p = (p for (p, q, label) in inv_a_p if label == a)
        for (p1, p2) in partition.split(inv_a_p):
            for b in sigma:
                if (p1, b) in wait_set:
                    wait_set.add((p2, b))
                else:
                    smaller = partition.select_smaller(p1, p2)
                    wait_set.add((smaller, b))

    return partition


sink = float("inf")


def add_sink_vertex(G, sigma=None):
    if sigma is None:
        sigma = alphabet(G)
    G.add_node(sink)
    for q in G:
        ol = out_labels(G, q)
        diff = [a for a in sigma if a not in ol]
        for a in diff:
            G.add_edge(q, sink, label=a)

    for a in sigma:
        G.add_edge(sink, sink, label=a)


@contextmanager
def sink_context(G, sigma=None):
    add_sink_vertex(G, sigma)
    try:
        yield
    finally:
        G.remove_node(sink)


def get_follower_equivalences(G):
    """Gets the follower-equivalences of `G`.

    In a deterministic labeled graph `G`, two vertices `p`
    and `q` are :dfn:`follower-equivalent` if there is no
    word `w` such that ``dot(G, p, w)`` and ``dot(G, q, w)``
    are not both None.

    Returns a pair of dicts representing a partition of
    `G`, where two states are follower-equivalent iff they
    are in the same part of the partition.

    Parameters
    ----------
    G : deterministic label graph

    Returns
    -------
    parts : dict
     maps each part number to a part of the partition
    part_lookup : dict
     maps each vertex of `G` to a part number

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 3, label="a")
    >>> G.add_edge(2, 3, label="a")
    >>> G.add_edge(1, 1, label="b")
    >>> G.add_edge(2, 2, label="b")
    >>> G.add_edge(3, 3, label="b")
    >>> parts, part_lookup = sd.get_follower_equivalences(G)
    >>> parts
    {0: {1, 2}, 2: {3}}
    >>> part_lookup
    {1: 0, 3: 2, 2: 0}

    See Also
    --------
    :func:`is_follower_separated`
    :func:`reduce`

    """
    with sink_context(G):
        partition = hopcroft(G, [sink])

    sink_p = partition.part_lookup[sink]
    del partition.parts[sink_p]
    del partition.part_lookup[sink]
    return partition.parts, partition.part_lookup


def is_follower_separated(G):
    """Returns True iff `G` is follower-separated.

    A deterministic labeled graph `G` if no two distinct
    vertices are follower-equivalant.

    Parameters
    ----------
    G : deterministic labeled graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 3, label="a")
    >>> G.add_edge(2, 3, label="a")
    >>> G.add_edge(1, 1, label="b")
    >>> G.add_edge(2, 2, label="b")
    >>> G.add_edge(3, 3, label="b")
    >>> sd.is_follower_separated(G)
    False

    See Also
    --------
    :func:`get_follower_equivalences`
    :func:`reduce`

    """
    parts, _ = get_follower_equivalences(G)
    return all(len(part) == 1 for part in parts.values())


def pair_shrink_graph(G):
    sigma = alphabet(G)
    S = nx.MultiDiGraph()
    S.add_node(sink)
    for pair in combinations(G, 2):
        pair = iset(pair)
        for a in sigma:
            res = idot(G, pair, [a])
            if len(res) > 0:
                if len(res) == 1:
                    res = sink
                S.add_edge(pair, res, label=a)
    return S


def returns_word(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        as_str = kwargs.pop("as_str", False)
        w = f(*args, **kwargs)
        if as_str:
            return "".join(w)
        else:
            return w
    return decorated


@returns_word
def get_path_label(G, path):
    label = []
    for p, q in zip(path, islice(path, 1, None)):
        edge = first(G[p][q].values())
        label.append(edge["label"])
    return label


@returns_word
def extend_to_synchronizing_word(G, w):
    """Attempt to extend `w` to a synchronizing word in `G`.

    Parameters
    ----------
    G : deterministic labeled graph
    w : word

    Returns
    -------
    u : word or None
     a synchronizing word `u` extending `w` if successful,
     None otherwise
    """
    S = pair_shrink_graph(G)
    paths = nx.shortest_path(S, target=sink)
    w = list(w)
    X = idot(G, G, w)
    while len(X) > 1:
        pq = iset(islice(X, 2))
        if pq in paths:
            u = get_path_label(S, paths[pq])
            X = idot(G, X, u)
            w.extend(u)
        else:
            return None
    return w


@returns_word
def find_synchronizing_word(G):
    """Search for a synchronizing word in `G`.

    A :dfn:`synchronizing word` in `G` is a word `w` such that
    ``len(idot(G, G, w)) == 1``.

    If `G` is follower-separated, a synchronizing word
    always exists and one is always returned.

    If `G` is irreducible or fully deterministic,
    then the return value is not None iff
    there is a synchronizing word.

    Otherwise, the return value being None does not imply
    no synchronizing word exists in `G`.

    Parameters
    ----------
    G : determinstic labeled graph

    Returns
    -------
    word or None
     a synchronizing word in `G`, None if search was unsuccessful

    See Also
    --------
    :func:`idot`

    """
    return extend_to_synchronizing_word(G, [])


def find_synchronizing_word2(G):
    X = iset(G)
    w = []
    with sink_context(G):
        paths = nx.shortest_path(G, target=sink)
        while len(X) > 1:
            for q in X:
                u = get_path_label(G, paths[q])
                Y = idot(G, X, u)
                if len(Y) > 1:
                    X = iset(q for q in Y if q != sink)
                    w.extend(u)
                    break
    return w


def reduce(G):
    """Compute the reduction of `G`.

    The :dfn:`reduction` of `G` is the graph resulting
    from  merging follower-equivalant states in `G`.

    Parameters
    ----------
    G : deterministic labeled graph

    Returns
    -------
    deterministic labeled graph
     the reduction of `G`

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 3, label="a")
    >>> G.add_edge(2, 3, label="a")
    >>> G.add_edge(1, 1, label="b")
    >>> G.add_edge(2, 2, label="b")
    >>> G.add_edge(3, 3, label="b")
    >>> list(sd.reduce(G).edges(data="label"))
    [({1, 2}, {3}, 'a'), ({1, 2}, {1, 2}, 'b'), ({3}, {3}, 'b')]

    See Also
    --------
    :func:`get_follower_equivalences`
    :func:`is_follower_separated`

    """
    parts, part_lookup = get_follower_equivalences(G)
    # convert the parts into iset
    for k in parts:
        parts[k] = iset(parts[k])

    Gr = nx.MultiDiGraph()
    for (p, q, a) in G.edges(data="label"):
        p_set = parts[part_lookup[p]]
        q_set = parts[part_lookup[q]]
        # here, we are making sure not to add two edges of the
        # same label between the same two vertices
        # I believe follower-separation can be performed on
        # nondeterministic graphs, but since the only we calculate
        # follower equivalences is with deterministic graphs,
        # this check is sufficient
        if a not in out_labels(Gr, p_set):
            Gr.add_edge(p_set, q_set, label=a)

    return Gr


def asymmetric_shrink_graph(G, H):
    sigma = alphabet(G) | alphabet(H)
    S = nx.MultiDiGraph()
    S.add_node(sink)
    with sink_context(H, sigma=sigma):
        for p, q in product(G, H):
            if q == sink:
                continue
            for a in sigma:
                res_G = dot(G, p, [a])
                res_H = dot(H, q, [a])
                # node in G is alive
                # (no sink was added to G, which is why we check for None)
                if res_G is not None:
                    # node in H is alive
                    # (sink was added to H, which is why we check for sink)
                    if res_H != sink:
                        S.add_edge((p, q), (res_G, res_H), label=a)
                    # node in H was killed
                    else:
                        S.add_edge((p, q), sink, label=a)
                # otherwise, node in G is killed,
                # do not add any edge
    return S


@returns_word
def find_separating_word(G, H):
    """Search for a separting word between `G` and `H`.

    A :dfn:`separating word` between `G` and `H` is a word
    `w` such that ``len(idot(G, G, w)) > 0``
    and ``len(idot(H, H, w)) == 0``.

    If `G` is irreducible, then return value is not None
    iff there is a separating word between `G` and `H`.

    Otherwise, the return value being None does not imply
    there is no separating word between `G` and `H`.

    Parameters
    ----------
    G, H : deterministic labeled graphs

    Returns
    -------
    word or None
     a separating word between `G` and `H`, None if search was unsuccessful

    """
    S = asymmetric_shrink_graph(G, H)
    paths = nx.shortest_path(S, target=sink)
    p = first(G)
    X = iset(H)
    w = []
    while len(X) != 0:
        q = first(X)
        if (p, q) in paths:
            u = get_path_label(S, paths[(p, q)])
            p = dot(G, p, u)
            X = idot(H, X, u)
            w.extend(u)
        else:
            return None
    return w


def is_subshift(G, H):
    """Returns True iff the shift presented by `G` is contained
    in the shift presented by `H`.

    Requires `G` to be irreducible and `H` to be essential for
    the return value to be correct.

    Parameters
    ----------
    G : irreducible deterministic labeled graph
    H : essential deterministic labeled graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 1, label="0")
    >>> H = nx.MultiDiGraph()
    >>> H.add_edge(1, 1, label="1")
    >>> H.add_edge(1, 2, label="0")
    >>> H.add_edge(2, 1, label="0")
    >>> sd.is_subshift(G, H)
    True

    See Also
    --------
    :func:`find_separating_word`

    """
    return find_separating_word(G, H) is None


@returns_word
def find_synchronizing_word_in_component(G, C):
    C_complement = [q for q in G if q not in C]
    G_C = G.subgraph(C).copy()
    G_C_complement = G.subgraph(C_complement).copy()
    w = find_separating_word(G_C, G_C_complement)
    if w is None:
        return None

    return extend_to_synchronizing_word(G_C, w)


def _sync_words_ics(G):
    condensate = nx.condensation(G)
    initial_components = [
        iset(C) for (Cp, C) in condensate.nodes(data="members")
        if not condensate.in_edges(Cp)
    ]
    for C in initial_components:
        yield (C, find_synchronizing_word_in_component(G, C))


def is_synchronizing(G):
    """Returns True iff `G` is synchronizing.

    A deterministic graph `G` is synchronizing if for each
    vertex `q` in `G` there is a word `w` such that
    ``idot(G, G, w) == iset([q])``.

    Parameters
    ----------
    G : deterministic labeled graph

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(1, 1, label="a")
    >>> G.add_edge(1, 2, label="b")
    >>> G.add_edge(2, 2, label="a")
    >>> sd.is_synchronizing(G)
    False
    >>> G.add_edge(1, 1, label="c")
    >>> sd.is_synchronizing(G)
    True

    """
    return all(w is not None for (_, w) in _sync_words_ics(G))


def is_label_isomorphic_fs(G, H):
    return find_label_isomorphism_fs(G, H) is not None


def find_label_isomorphism_fs(G, H):
    Gp = nx.relabel_nodes(G, lambda x: (x, 1))
    Hp = nx.relabel_nodes(H, lambda x: (x, 2))
    GH = nx.union(Gp, Hp)
    parts, _ = get_follower_equivalences(GH)
    mapping = {}
    for part in parts.values():
        if len(part) != 2:
            return None
        (qG, _), (qH, _) = sorted(part, key=lambda x: x[1])
        mapping[qG] = qH

    return mapping


def are_shifts_equal_sync(G, H):
    """Returns True iff the shift presnted by `G` is equivalent
    to the shift presented by `H`.

    Requires `G` and `H` to be synchronizing for the return value
    to be correct.

    Parameters
    ----------
    G, H : synchronizing deterministic graphs
    """
    return is_label_isomorphic_fs(reduce(G), reduce(H))
