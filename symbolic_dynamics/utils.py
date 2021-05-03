class iset(frozenset):
    def __repr__(self):
        return "{" f"{', '.join(repr(i) for i in self)}" "}"


def first(it):
    return next(iter(it))
