"""
    Interface to a corpus of Named Entities tagged

    Needed functionality:
        load corpus from pickle
        get sequence of part of speech taggers
        split corpus
        vault the split
        support shelving of the corpus
"""
from nltk_corpora_interface import corpus, corpus_data, corpus_datum
from nltk import tokenize
import itertools

class TreebankNER(corpus):

    def __init__(self, training, dev, test=None):
        if test is None:
            test = dev
        self.training = training
        self.dev = dev
        self.test = test

    @classmethod
    def make(filename):
        pass

    @classmethod
    def clean(cls, raw_filename):
        tok = lambda x: tokenize.word_tokenize(x)
        with open(raw_filename,"rb") as fp:
            lines = [x.replace("\n", "") for x in fp.readlines()]
        data = []
        for line in lines:
            notner = "NO"
            tokens = tok(line)
            datum = []
            curnertag = notner
            i_tokens = iter(tokens)
            for token in i_tokens:
                if "<" in token:
                    for nertag in itertools.islice(i_tokens,2):
                        if nertag[0] == "/" and ">" not in nertag:
                            curnertag = notner
                        elif ">" not in nertag:
                            curnertag = nertagb
                    continue
                if token == "." and curnertag is not notner:
                    tokj,tagj = datum[-1]
                    datum[-1] = (tokj+".",tagj)
                else:
                    datum.append((token,curnertag))
            if any([1 for x in datum if x[1] == "lost"]):
                print "gotit"
                print line
                continue
            data.append(datum)
        return TreebankNER.initial_split(data)

    @classmethod
    def from_vault(cls, vault_information):
        pass

    @classmethod
    def initial_split(cls, data):
        return data
