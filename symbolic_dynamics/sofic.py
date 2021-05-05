"""Functions and algorithms for sofic shifts.
"""

"""
Most functions in this module expected an edge-lableled
:class:`networkx.MultiDiGraph`, which we will just 
refer to as a *labeled graph*. More specifically,
a labeled graph is a :class:`networkx.MultiDiGraph` where 
each edge has the attribute ``label``. (:func:`is_labeled`).
Labels should be hashable.

Another 
"""


import networkx as nx
from symbolic_dynamics.utils import iset, first
from itertools import combinations, islice, product
import random
from contextlib import contextmanager


def dot(G, q, w):
    """Computes the transition action of `w` in `G` on `q`.

    In a deterministic labeled graph `G`, any path starting at a 
    given vertex is uniquely determinied by its sequence of labels.
    If there is a path labeled `w` starting at `q` in `G`, then 
    the transition action of `w` in `G` on `q` is defined to be 
    the vertex that path ends at. Otherwise, if there is no such 
    path, then the transtion action of `w` in `G` on `q` is
    defined to be ``None``.

    :param G: a deterministic labeled graph 
    :param q: a vertex in `G`
    :param w: a word

    **Example**

    >>> G = nx.MultiDiGraph()
    >>> nx.add_path(G, range(5), label="a")
    >>> sd.dot(G, 0, "aaaa")
    4
    >>> sd.dot(G, 1, "aaaa")
    None

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
    of the iterable `it`, returning an :class:`iset` of the non-``None``
    results.

    :param G: a deterministic labeled graph
    :param it: an iterable of vertices in `G`
    :param w: a word

    **Example**

    >>> G = nx.MultiDiGraph()
    >>> nx.add_path(G, range(5), label="a")
    >>> sd.idot(G, [0, 1], "aaa")
    {3, 4}
    >>> sd.idot(G, [0, 1], "aaaa")
    {4}
    >>> sd.idot(G, [0, 1], "aaaaa")
    {}

    """
    x = (dot(G, q, w) for q in it)
    x = (q for q in x if q is not None)
    return iset(x)


def is_labeled(G):
    """Returns ``True`` iff every edge in the graph `G` has the attribute ``label``.
    
    :param G: a graph
    """
    return all("label" in data for _, _, data in G.edges(data=True))


def alphabet(G):
    """Returns an :class:`iset` of labels appearing in `G`.

    :param G: a labeled graph
    """
    return iset(label for (_, _, label) in G.edges(data="label"))


def out_labels(G, q):
    """Returns a list of each of the labels appearing on the edges
    starting at `q` in `G`

    :param G: a labeled graph
    :param q: a vertex in `G`
    """
    return [label for (_, _, label) in G.out_edges(q, data="label")]


def is_extensible(G, q):
    return bool(G.out_edges(q))


def is_coextensible(G, q):
    return bool(G.in_edges(q))


def is_stranded(G, q):
    """Returns ``True`` iff `q` is stranded in `G`.

    A vertex `q` in `G` is *stranded* if there is no 
    edge starting at `q` or if there is no edge ending at `q`.

    :param G: a graph
    :param q: a vertex in `G`
    """
    return not (is_extensible(G, q) and is_coextensible(G, q))


def is_essential(G):
    """Returns ``True`` iff `G` is essential.

    A graph `G` is *essential* if no vertex is stranded.

    :param G: a graph
    """
    return all(not is_stranded(G, q) for q in G)


def make_essential(G):
    """Modifies `G` by removing vertices in `G` that do not lie 
    on bi-infinite paths. The resulting graph is the maximal essential
    subgraph of the original graph.

    :param G: a graph
    """
    nonextensible = [q for q in G if not is_extensible(G, q)]
    while True:
        frontier = iset(q for (q, _) in G.in_edges(nonextensible))
        G.remove_nodes_from(nonextensible)
        nonextensible = [q for q in frontier if not is_extensible(G, q)]
        if not nonextensible:
            break

    noncoextensible = [q for q in G if not is_coextensible(G, q)]
    while True:
        frontier = iset(q for (_, q) in G.out_edges(noncoextensible))
        G.remove_nodes_from(noncoextensible)
        noncoextensible = [q for q in frontier if not is_coextensible(G, q)]
        if not noncoextensible:
            break


def is_deterministic(G):
    """Returns ``True`` iff `G` is deterministic.

    A labeled graph `G` is :dfn:`deterministic` if every 
    edge starting at a given vertex is labeled uniquely. 

    :param G: a labeled graph

    **Example**

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
    """Returns ``True`` iff `G` is fully deterministic.

    A labeled graph `G` is :dfn:`fully deterministic` if 
    `G` is :ref:`deterministic <deterministic>` and for every vertex `q` in `G`,
    the set of labels of edges starting at `q` is equal
    the set of labels appearing in `G`.
     
    :param G: a labeled graph

    **Example**

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

    More specifically, if ``G`` is the result of ``from_partial_fns(pfns)``, 
    then ``G`` has the following property:

    >>> for (a, pfn) in pfns.items():
    ...     for q in G:
    ...         if q in pfn:
    ...              assert dot(G, q, [a]) == pfn[q]
    ...         else:
    ...              assert dot(G, q, [a]) is None

    :param pfns: dict of dicts; see above

    **Example**

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
    return from_partial_fns({str(i): random_partial_fn(n) for i in range(m)})


def random_deterministic_graph_with_props(n, m, props):
    while True:
        G = random_deterministic_graph(n, m)
        if all(p(G) for p in props):
            return G


def is_irreducible(G):
    """Returns True iff `G` is irreducible

    A graph is :dfn:`irreducible` if for any two vertices in `G`,
    there is a path between them. Synonymous with *strongly connected*.

    :param G: a graph
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
                marks[p] = []
            marks[p].append(element)

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
    with sink_context(G):
        partition = hopcroft(G, [sink])

    sink_p = partition.part_lookup[sink]
    del partition.parts[sink_p]
    del partition.part_lookup[sink]
    return partition


def is_follower_separated(G):
    partition = get_follower_equivalences(G)
    return all(len(part) == 1 for part in partition.parts.values())


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
    partition = get_follower_equivalences(G)
    # convert the parts into iset
    for k in partition.parts:
        s = partition.parts[k]
        partition.parts[k] = iset(s)

    Gr = nx.MultiDiGraph()
    for (p, q, a) in G.edges(data="label"):
        p_set = partition.parts[partition.part_lookup[p]]
        q_set = partition.parts[partition.part_lookup[q]]
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
    return all(w is not None for (_, w) in _sync_words_ics(G))


def is_label_isomorphic_fs(G, H):
    return find_label_isomorphism_fs(G, H) is not None


def find_label_isomorphism_fs(G, H):
    Gp = nx.relabel_nodes(G, lambda x: (x, 1))
    Hp = nx.relabel_nodes(H, lambda x: (x, 2))
    GH = nx.union(Gp, Hp)
    parts = get_follower_equivalences(GH).parts
    mapping = {}
    for part in parts.values():
        if len(part) != 2:
            return None
        (qG, _), (qH, _) = sorted(part, key=lambda x: x[1])
        mapping[qG] = qH

    return mapping
