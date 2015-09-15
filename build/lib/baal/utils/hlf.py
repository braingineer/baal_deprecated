from baal.utils.data_structures.flyweight import Flyweight


class gensym(Flyweight):
    __slots__ = ["sym_generator", "headword", "symbol","number_inst", "gorn"]
    sym_generator = ("g%d" % i for i in xrange(10**10))

    def __init__(self, headword, gorn):
        try:
            self.number_inst += 1
        except:
            self.headword = headword
            self.symbol = next(self.sym_generator)
            self.number_inst = 1
            self.gorn = gorn

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return repr(self.__str__())

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def __hash__(self):
        return hash(self.headword)+hash(self.gorn)+hash(self.symbol)

    def __eq__(self, other):
        if other.symbol == self.symbol:
            assert other.headword == self.headword
        return other.symbol == self.symbol

    def str(self):
        return self.symbol

    def verbose(self):
        return self.symbol + "<%s>" % self.gorn



class unboundgensym(Flyweight):
    __slots__ = ["sym_generator", "headword", "symbol","number_inst", "gorn"]
    sym_generator = ("X%d" % i for i in xrange(10**10))

    def __init__(self, gorn):
        try:
            self.number_inst += 1
        except:
            self.symbol = self.headword = next(self.sym_generator)
            self.number_inst = 1
            self.gorn = gorn

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return repr(self.__str__())

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def __hash__(self):
        return hash(self.symbol)+hash(self.gorn)

    def __eq__(self, other):
        return other == self.symbol

    def str(self):
        return self.symbol

    def verbose(self):
        return self.symbol + "<%s>" % self.gorn
