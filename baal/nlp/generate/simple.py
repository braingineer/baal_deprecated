"""
A simple generation algorithm

Assumes the following:
    1. grammar consists of baal.utils.data_structures.trees
    2. start symbol is "S"
    3. it's my grammar formalism

Notes:
    1. insert direction == right :
        this means that the tree wants to be the last child.
        its insertion point is on its left side
        its inserting into the tree on that tree's right side
        so, a prepositional phrase "at" will be a left inserting tree

"""

from baal.nlp.grammars import protogist
from distance import levenshtein # in pypi
try:
    import cpickle as pickle
except:
    import pickle

class consts:
    INS = "insertion"
    SUB = "sub"
    S   = "S"
    NP  = "NP"
    VP  = "VP"

class generator(object):

    def __init__(self, grammar):
        self.grammar = grammar
        self.init()

    def init(self):
        # grammar by root
        self.all_root_syms = set([x.tree.symbol for x in self.grammar])
        self.grammar_by_root = {root_sym:[] for root_sym in self.all_root_syms}
        for entry in self.grammar:
            self.grammar_by_root[entry.tree.symbol].append(entry)

        # grammar by type
        self.grammar_by_type = {consts.INS: [x for x in self.grammar
                                             if x.tree.adjunct],
                                consts.SUB: [x for x in self.grammar
                                             if not x.tree.adjunct]}

        # grammar by type and root symbol
        self.grammar_by_roottype = {consts.INS: {root:[x for x in self.grammar
                                                         if x.tree.adjunct
                                                         and x.tree.symbol==root]
                                                       for root in self.all_root_syms},
                                    consts.SUB: {root:[x for x in self.grammar
                                                         if not x.tree.adjunct
                                                         and x.tree.symbol==root]
                                                       for root in self.all_root_syms}}

    def bfs(self, max_depth=2):
        """
            core bfs algorithm; constructs sentences based on grammar
            currently, constructs based on nothing else but depth
            later, will construct to string match or fulfull goals

            procedure:
                find all trees that start with S
                find all insertion trees that can join to it
                find all substitution trees that can join to it
                consider this the frontier

                do breadth first search, reporting at the number of candidates at every step
        """
        depth = 0
        horizon = {depth:[]}
        all_S_entries = self.grammar_by_root[consts.S]
        horizon[depth].extend(all_S_entries)
        edge_conditionals = (True, True)

        while depth <= max_depth:
            print("{} entries at depth {}".format(len(horizon[depth]), depth))
            horizon[depth+1] = []
            for horizon_entry in horizon[depth]:
                for ins_entry in self.grammar_by_type[consts.INS]:
                    for new_entry in horizon_entry.combine(ins_entry, edge_conditionals):
                        horizon[depth+1].append(new_entry)
                for sub_entry in self.grammar_by_type[consts.SUB]:
                    for new_entry in horizon_entry.combine(sub_entry, edge_conditionals):
                        horizon[depth+1].append(new_entry)
            depth += 1

    def set_search(self, word_list, max_depth=10):
        """
            Search through tree combinations while only choosing resources which
            move towards matching the word_set.
        """
        depth = 0
        horizon = {depth:set()}
        edge_conditionals = (True, True)

        def consistent(entry, word_list=word_list):
            new_word_list = []
            for address, subtree in sorted(entry.lexical, key=lambda (addr,subt): addr):
                new_word = subtree.symbol
                if new_word not in word_list:
                    return False

                new_word_list.append(new_word)
            dist = levenshtein(new_word_list,
                                    word_list) - abs(len(new_word_list)-len(word_list))
            return dist == 0

        def successful(entry, word_list=word_list):
            new_word_list = []
            for address, subtree in sorted(entry.lexical, key=lambda (addr,subt): addr):
                new_word = subtree.symbol
                if new_word not in word_list:
                    return False

                new_word_list.append(new_word)
            return levenshtein(new_word_list, word_list) == 0

        #consistent = lambda entry: len(set([subtree.symbol
        #                                    for address,subtree in entry.lexical])
        #                               -word_set) == 0

        all_S_entries = self.grammar_by_root[consts.S]
        horizon[depth].update(entry for entry in all_S_entries if consistent(entry))

        good_subs = [sub_entry for sub_entry in self.grammar_by_type[consts.SUB]
                     if consistent(sub_entry)]
        good_ins = [ins_entry for ins_entry in self.grammar_by_type[consts.INS]
                    if consistent(ins_entry)]

        successes = []

        while depth <= max_depth:
            print("{} entries at depth {}".format(len(horizon[depth]), depth))
            horizon[depth+1] = set()
            mod10perc = int(len(horizon[depth]) * 0.1)
            for i, horizon_entry in enumerate(horizon[depth]):
                if successful(horizon_entry):
                    print("Found a success")
                    successes.append(horizon_entry)
                    continue
                if  i % mod10perc == 0 and i > 0:
                    print("{:.2%} percent complete".format(float(i)/len(horizon[depth])))
                #print "Starting insertion entries"
                for ins_entry in good_ins:
                    for new_entry in horizon_entry.combine(ins_entry, edge_conditionals):
                        if consistent(new_entry):
                            horizon[depth+1].add(new_entry)
                #print "Starting substitution entries"
                for sub_entry in good_subs:
                    for new_entry in horizon_entry.combine(sub_entry, edge_conditionals):
                        if consistent(new_entry):
                            horizon[depth+1].add(new_entry)
            print("here's a sample:")
            for anexample in list(horizon[depth+1])[:30]:
                print anexample
            depth += 1
        print("Found {} successes".format(len(successes)))
        print("Dumping to a file for playing around")
        with open("testing.pkl", "wb") as fp:
            pickle.dump(successes, fp)


def test0():
    print("Making grammars")
    nlg = generator(protogist.make("8k"))
    print("Starting breadth first search")
    nlg.bfs()

def test1():
    print("Making grammars")
    nlg = generator(protogist.make("8k"))
    print("Starting breadth first search")
    nlg.set_search("A man is snowboarding down a skislope .".split(" "))

if __name__ == "__main__":
    test1()


