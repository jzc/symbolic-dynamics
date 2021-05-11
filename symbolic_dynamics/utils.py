"""Utilities and miscellaneous functions."""
class iset(frozenset):
    """An immutable (and hashable) set.
    
    This class is just a wrapper around :class:`frozenset` 
    with a shorter name and modifying :meth:`__repr__` to 
    look like :meth:`set.__repr__`.

    Examples
    --------
    >>> s = iset([1,2])
    >>> d = {s: 3}
    >>> d
    {{1, 2}: 3}
    >>> d[s]
    3

    """
    def __repr__(self):
        return "{" f"{', '.join(repr(i) for i in self)}" "}"


def first(it):
    """Get the first element of an iterable."""
    return next(iter(it))
