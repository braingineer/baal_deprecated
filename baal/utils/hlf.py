from weakref import WeakValueDictionary


class gensym(object):
    __slots__ = ["_instances","__weakref__", "sym_generator", "head", "symbol","number_inst", "pos_symbol"]
    _instances = WeakValueDictionary()
    sym_generator = ("g%d" % i for i in xrange(10**10))

    def __new__(cls, *args, **kargs):
        if 'address' not in kargs:
            raise ValueError("pass addresses")
        if 'head' not in kargs:
            raise ValueError("pass head")

        base = kargs['address'][:-1]
        key = (cls, kargs['head'], tuple(base))
        if key not in cls._instances:
            return cls._instances.setdefault(key,
                                             super(gensym, cls).__new__(cls))
        else:
            newkey = (cls, kargs['head'], tuple(kargs['address']))
            return cls._instances.setdefault(newkey,
                                             cls._instances[key])
        """
        for element in kargs['address']:
            base.append(element)
            try:
                key = (cls, kargs['head'], tuple(base))
                if key not in cls._instances:
                    continue
                else:
                    return cls._instances[key]
            except TypeError as e:
                print key
                raise e
        return cls._instances.setdefault(key,
                                         super(gensym,cls).__new__(cls))
        """

    def __init__(self, head="", address="", symbol=""):
        try:
            self.number_inst += 1
        except:
            self.head = head
            self.symbol = next(self.sym_generator)
            self.number_inst = 1
            self.pos_symbol = symbol

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return repr(self.__str__())

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def __hash__(self):
        return hash(self.head)+hash(self.symbol)

    def __eq__(self, other):
        if other.symbol == self.symbol:
            assert other.head == self.head
        return other.symbol == self.symbol

    def str(self):
        return self.symbol

    def verbose(self):
        return self.symbol



class unboundgensym(gensym):
    __slots__ = ["sym_generator", "head", "symbol","number_inst", "pos_symbol"]
    sym_generator = ("X%d" % i for i in xrange(10**10))

    def __init__(self, head="", address=None, symbol=""):
        try:
            self.number_inst += 1
        except:
            self.symbol = self.head = next(self.sym_generator)
            self.number_inst = 1
            self.pos_symbol = symbol

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return repr(self.__str__())

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        return other.symbol == self.symbol

    def str(self):
        return self.symbol

    def verbose(self):
        return self.symbol
