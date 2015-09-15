def _articles(word):
    ret = ["(NP (DT %s) (NN))" % word]
    ret += ["(NP (DT %s) (NP))" % word]
    return ret

def _nouns(word):
    ret = ["(NP (NN %s))" % word]
    ret += ["(NN %s)" % word]
    return ret


def _iverbs(word):
    ret = ["(S (NP) (VP (V %s)))" % word]
    return ret

def _tverbs(word):
    ret = ["(S (NP) (VP (V %s) (NP)))" % word]
    ret += ["(S (VP (V %s) (NP)))" % word]
    ret += ["(S (VP (V %s) (PP)))" % word]
    ret += ["(S (NP) (VP (V %s) (PP)))" % word]
    return ret

def _overbs(word):
    # I don't know what kind of verbs these are... so gen all
    ret =  _iverbs(word)+_tverbs(word)
    return ret

def _adjective(word):
    ret = ["(NP* (ADJ %s))" % word]
    ret += ["(*NP (ADJ %s))" % word]
    return ret

def _prepositions(word):
    ret = ["(*NP (PP (IN %s) (NP)))" % word]
    ret += ["(*VP (PP (IN %s) (NP)))" % word]
    ret += ["(PP (IN %s) (NP))" % word]
    return ret

def _adverbs(word):
    ret = ["(VP* (ADVP %s))" % word]
    ret += ["(*VP (ADVP %s))" % word]
    return ret

def _pronouns(word):
    ret = ["(NN (PRP %s))" % word]
    return ret

# /// Potentially two noun lists ///
def _nouns1(word):
    pass

def _nouns2(word):
    pass
# /// End potential two noun lists ///

def generate_rules(word_type, word):
    mapper = {"articles": _articles,
              "nouns1": _nouns,
              "nouns2": _nouns,
              "iverbs": _iverbs,
              "tverbs": _tverbs,
              "overbs": _overbs,
              "adjective": _adjective,
              "prepositions": _prepositions,
              "adverbs": _adverbs,
              "pronouns": _pronouns}
    func = mapper[word_type]
    return func(word)
