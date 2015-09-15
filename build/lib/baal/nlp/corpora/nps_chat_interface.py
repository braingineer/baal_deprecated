"""
Inherit things from the nltk_corpora_interface
"""
import nltk
from nltk.corpus import nps_chat
from types import StringType
from copy import deepcopy
from operator import mul
from baal.utils.vocabulary import Vocabulary
from baal.utils.general import flatten, get_ind


class nps(object):

    def __init__(self):
        pass

    @staticmethod
    def get_session(fileid):
        # This will return a session
        return nps_chat.xml(fileid)  # @UndefinedVariable  nps

    @staticmethod
    def make_sessions():
        fileids = nps_chat.fileids()  # @UndefinedVariable
        for fileid in fileids:
            yield nps_data.fromxml(nps.get_session(fileid), fileid)


class nps_data(object):

    def __init__(self, posts, name=None):
        if not name:
            name = "Why am I nameless"
        self.name = name
        self.posts = posts
        self.filtered_posts = deepcopy(posts)
        self.toggle = self.enumerate
        self.filters = []
        self.lim = (0, len(self.filtered_posts))

        self.toggler('action')
        self.action_vocab = Vocabulary.from_iterable(x[1] for x in self)
        self.toggler('pos')
        self.pos_vocab = Vocabulary.from_iterable(
            flatten(get_ind(x, 1) for x in self))
        self.word_vocab = Vocabulary.from_iterable(
            flatten(get_ind(x, 0) for x in self))

        self.toggler('enum')

    @classmethod
    def fromxml(cls, session_xml, fileid):
        root = session_xml
        posts = [nps_post(item)
                 for item in root.getchildren()[0].getchildren()]
        return cls(posts, fileid)

    @classmethod
    def make(cls, posts, name):
        return cls(posts, name)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [(i, self.filtered_posts[i]) for i in xrange(*k.indices(len(self.filtered_posts)))]
        elif isinstance(k, StringType):
            return [p for p in self.filtered_posts if p.post_class == k]
        elif isinstance(k, int):
            return self.filtered_posts[k]
        elif isinstance(k, tuple):
            return self.yield_subset(*k)
        else:
            raise TypeError, 'unable to comply'

    def __setitem__(self, k, v):
        raise TypeError("Setting an item doesn't make sense in this context")

    def __iter__(self):
        return self.toggle()

    def __len__(self):
        return len(self.filtered_posts)

    def yield_subset(self, llim, ulim):
        self.lim = (llim, ulim)
        for p in self:
            yield p

    def add_filter(self, filter):
        self.filters.append(filter)
        self._filter_data()
        self.lim = (self.lim[0], len(self.filtered_posts))

    def filter_system_posts(self):
        self.add_filter(lambda x: x.post_class != "System")

    def _filter_data(self):
        keep = lambda s: reduce(
            mul, [1] + [1 if f(s) else 0 for f in self.filters])
        self.filtered_posts = [post for post in self.posts if keep(post)]

    def toggler(self, tog):
        self.toggle = {'enum': self.enumerate, 'pos':
                       self.pos_data_genesis, 'action': self.action_data_genesis}[tog]

    def enumerate(self):
        for post in self.posts[self.lim[0]:self.lim[1]]:
            yield post

    def pos_data_genesis(self):
        """
        Yields data in the form [(token,tag),...]
        Each yield should be considered a 'chain' (a full sequence)
        """
        for post in self.filtered_posts[self.lim[0]:self.lim[1]]:
            yield [(x['word'], x['pos']) for x in post.terminals]

    def action_data_genesis(self):  # dialogue act is action
        """
        Yields data in the form (token,tag)
        Each yield should be considered a single observation in a chain
        The token is the post object, so feature extraction should be performed.
        """
        for post in self.posts[self.lim[0]:self.lim[1]]:
            yield (post, post.post_class)


class nps_post(object):

    def __init__(self, post_xml):
        rind = lambda x, y: x.rindex(y)
        spliton = lambda x, s: [x[:rind(x, s)], x[rind(x, s):]]

        self.post_class, self.post_user = [x[1] for x in post_xml.items()]
        self.tag, self.post_user = spliton(self.post_user, 'User')

        remove_tag = lambda s: s.replace(self.tag, "")
        self.terminals = [{k: remove_tag(v) for k, v in term.items(
        )} for term in post_xml.getchildren()[0].getchildren()]
        self.text = remove_tag(post_xml.text)

    def __str__(self):
        return "%s commits %s: ''%s''" % (self.post_user, self.post_class, self.text)

    def __repr__(self):
        return self.__str__()
