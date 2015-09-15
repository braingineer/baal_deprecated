"""
Grammar Application Protocol Interface

This will serve as the template for grammars

@author bcmcmahan
@date 5/16/2015
"""
from baal.utils.data_structures import trees
from baal.nlp import lexicon

def clean(x):
    if len(x) == 0:
        return []
    return [y.strip() for y in x.split("\n") if len(y.strip())>0]

def make(rules,cleaned=False):
    rules = clean(rules) if not cleaned else rules
    return [trees.Entry.make(bracketed_string=x) for x in rules]

class Grammar(object):
    def __init__(self, rules=[]):
        forest = make(rules)
        self.indexed_forest = {}
        for entry in forest:
            lex_item = entry.tree.head
            self.indexed_forest.setdefault(lex_item,[]).append(entry)

    def from_word(self, word):
        for c_func in self._correction_generator():
            if self._from_word(c_func(word)):
                return self._from_word(c_func(word))
        raise KeyError, "I'm sorry, I don't have %s in my lexicon!" % word


    def _correction_generator(self):
        yield lambda x: x
        yield lambda x: x.lower()
        yield lambda x: x.upper()
        yield lambda x: x.capitalize()

    def _from_word(self, word):
        # see if its in our preferred grammar
        if word in self.indexed_forest.keys():
            return self.indexed_forest[word]

        # see if we can auto generate some trees
        auto_trees = self._checklists(word)
        if len(auto_trees) > 0:
            return auto_trees

        return False

    def _checklists(self, word):
        wordlists = lexicon.get_wordlists()
        # print wordlists.keys()
        auto_rules = []
        for word_type, word_set in wordlists.items():
            if word in word_set:
                auto_rules.extend(lexicon.generate_rules(word_type,word))
        # print auto_rules
        return make(auto_rules, True)


    def __getitem__(self, k):
        return self.from_word(k)



def tests():
    from jurafskymartin_L1 import L1_rules
    L1_grammar = Grammar(L1_rules)
    for tree in L1_grammar['book']:
        print tree

def tail_tests():
    blank_grammar = Grammar()
    for tree in blank_grammar['aardvark']:
        print tree

    for tree in blank_grammar['box']:
        print tree

if __name__ == "__main__":
    tests()
    tail_tests()


