import networkx as nx
from symbolic_dynamics.utils import iset, first
from itertools import combinations, islice
import random
from contextlib import contextmanager


def dot(G, q, w):
    if q is None:
        return None

    for a in w:
        delta = {ell: p for (_, p, ell) in G.out_edges(q, data="label")}
        if a not in delta:
            return None
        q = delta[a]

    return q


def idot(G, it, w):
    x = (dot(G, q, w) for q in it)
    x = (q for q in x if q is not None)
    return iset(x)


def alphabet(G):
    return iset(label for (_, _, label) in G.edges(data="label"))
    
    
def out_labels(G, q):
    return [label for (_, _, label) in G.out_edges(q, data="label")]


def is_extensible(G, q):
    return bool(G.out_edges(q))


def is_coextensible(G, q):
    return bool(G.in_edges(q))

    
def is_stranded(G, q):
    return not (is_extensible(G, q) and is_coextensible(G, q))


def is_essential(G):
    return all(not is_stranded(G, q) for q in G)


def make_essential(G):
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
    for q in G:
        ol = out_labels(G, q)
        if len(ol) != len(set(ol)):
            return False
    return True


def is_fully_deterministic(G):
    n = len(out_labels(G, first(G)))
    for q in G:
        ol = out_labels(G, q)
        m = len(ol)
        if not (len(set(ol)) == m and n == m):
            return False
    return True


def from_partial_fns(pfns):
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

    return partition.parts

sink = float("inf")

def add_sink_vertex(G):
    sigma = alphabet(G)
    G.add_node(sink)
    for q in G:
        ol = out_labels(G, q)
        for a in sigma.difference(ol):
            G.add_edge(q, sink, label=a)

    for a in sigma:
        G.add_edge(sink, sink, label=a)

@contextmanager
def sink_context(G):
    Gs = add_sink_vertex(G)
    try:
        yield
    finally:
        G.remove_node(sink)

def get_follower_equivalences(G):
    with sink_context(G):
        partition = hopcroft(G, [sink])
    
    sink_p = first(k for k, v in partition.items() if sink in v)
    del partition[sink_p]
    return partition
        
def is_follower_separated(G):
    partition = get_follower_equivalences(G)
    return all(len(part) == 1 for part in partition.values())

        
def pair_shrink_graph(G):
    sigma = alphabet(G)
    S = nx.MultiDiGraph()
    for pair in combinations(G, 2):
        pair = iset(pair)
        for a in sigma:
            res = idot(G, pair, [a])
            if len(res) > 0:
                if len(res) == 1:
                    res = sink
                S.add_edge(pair, res, label=a)
    return S

def convert_str(w, as_str):
    if as_str:
        return "".join(w)
    else:
        return w

def get_path_label(G, path, as_str=False):
    label = []
    for p, q in zip(path, islice(path, 1, None)):
        edge = first(G[p][q].values())
        label.append(edge["label"])
    return convert_str(label, as_str)


def find_synchronizing_word(G, as_str=False):
    S = pair_shrink_graph(G)
    paths = nx.shortest_path(S, target=sink)
    X = iset(G)
    w = []
    while len(X) > 1:
        pq = iset(islice(X, 2))
        if pq in paths:
            u = get_path_label(S, paths[pq])
            X = idot(G, X, u)
            w.extend(u)
        else:
            return None
    return convert_str(w, as_str)

