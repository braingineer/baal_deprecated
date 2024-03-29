import os
from numpy.random import randint


class Vocabulary(object):
    """
    Taken from Tim Vieira & his Github: https://github.com/timvieira/arsenal


    Bijective mapping from strings to integers.

    Used for turning a bunch of string observations into integers (presumably for indexing and other fun things)

    >>> a = Vocabulary()
    >>> [a[x] for x in 'abcd']
    [0, 1, 2, 3]
    >>> map(a.lookup, range(4))
    ['a', 'b', 'c', 'd']

    >>> a.stop_growth()  # Thus, a growth stop means it will be nice about its closed-ness
    >>> a['e']

    >>> a.freeze() #freezing means harsher punishments for trying to go outside its domain
    >>> a.add('z')
    Traceback (most recent call last):
      ...
    ValueError: Vocabulary is frozen. Key "z" not found.

    >>> print a.plaintext()
    a
    b
    c
    d
    """

    def __init__(self, random_int=None):
        self._mapping = {}   # str -> int
        self._flip = {}      # int -> str; timv: consider using array or list
        self._i = 0
        self._frozen = False
        self._growing = True
        self._random_int = random_int   # if non-zero, will randomly assign
                                        # integers (between 0 and randon_int) as
                                        # index (possibly with collisions)

    def __repr__(self):
        return 'Vocabulary(size=%s,frozen=%s)' % (len(self), self._frozen)

    def freeze(self):
        self._frozen = True

    def stop_growth(self):
        self._growing = False

    @classmethod
    def from_iterable(cls, s):
        "Assumes keys are strings."
        inst = cls()
        for x in s:
            inst.add(x)
#        inst.freeze()
        return inst

    def keys(self):
        return self._mapping.iterkeys()

    def items(self):
        return self._mapping.iteritems()

    def imap(self, seq, emit_none=False):
        """
        Apply Vocabulary to sequence while filtering. By default, `None` is not
        emitted, so please note that the output sequence may have fewer items.
        """
        if emit_none:
            for s in seq:
                yield self[s]
        else:
            for s in seq:
                x = self[s]
                if x is not None:
                    yield x

    def map(self, seq, *args, **kwargs):
        return list(self.imap(seq, *args, **kwargs))

    def add_many(self, x):
        return [self.add(k) for k in x]

    def lookup(self, i):
        if i is None:
            return None
        #assert isinstance(i, int)
        return self._flip[i]

    def lookup_many(self, x):
        for k in x:
            yield self.lookup(k)

    def __contains__(self, k):
        #assert isinstance(k, basestring)
        return k in self._mapping

    def __getitem__(self, k):
        try:
            return self._mapping[k]
        except KeyError:
            #if not isinstance(k, basestring):
            #    raise ValueError("Invalid key (%s): only strings allowed." % (k,))
            if self._frozen:
                raise ValueError('Vocabulary is frozen. Key "%s" not found.' % (k,))
            if not self._growing:
                return None
            if self._random_int:
                x = self._mapping[k] = randint(0, self._random_int)
            else:
                x = self._mapping[k] = self._i
                self._i += 1
            self._flip[x] = k
            return x

    add = __getitem__

    def __setitem__(self, k, v):
        assert k not in self._mapping
        if self._frozen: raise ValueError("Vocabulary is frozen. Key '%s' cannot be changed")
        assert isinstance(v, int)
        self._mapping[k] = v
        self._flip[v] = k

    def __iter__(self):
        for i in xrange(len(self)):
            yield self._flip[i]

    def enum(self):
        for i in xrange(len(self)):
            yield (i, self._flip[i])

    def __len__(self):
        return len(self._mapping)

    def plaintext(self):
        "assumes keys are strings"
        return '\n'.join(self)

    @classmethod
    def load(cls, filename):
        if not os.path.exists(filename):
            return cls()
        with file(filename) as f:
            return cls.from_iterable(l.strip() for l in f)

    def save(self, filename):
        with file(filename, 'wb') as f:
            f.write(self.plaintext())
