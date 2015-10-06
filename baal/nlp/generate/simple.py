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
from baal.nlp.corpora import captions
from baal.utils.data_structures import trees
from baal.utils.general import cprint, cformat
from baal.nlp.semantics import simple_hlf
import baal
from baal.tests.loggers import shell_logs
from distance import levenshtein # in pypi

from collections import defaultdict


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
        self.grammar_by_root = defaultdict(lambda: [],
                                          {root_sym:[] for root_sym in self.all_root_syms})
        for entry in self.grammar:
            self.grammar_by_root[entry.tree.symbol].append(entry)

        # grammar by type
        self.grammar_by_type = defaultdict(lambda: [],
                                {consts.INS: [x for x in self.grammar
                                             if x.tree.adjunct],
                                consts.SUB: [x for x in self.grammar
                                             if not x.tree.adjunct]})

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
        print "wtf"
        depth = 0
        horizon = {depth:set()}
        edge_conditionals = (True, True)

        def consistent(entry, word_list=word_list):
            last_word = -1
            new_word_list = []
            sorted_lex = sorted(entry.lexical, key=lambda (addr,subt): addr)
            for i, (address, subtree) in enumerate(sorted_lex):
                new_word = subtree.symbol
                if new_word not in word_list[last_word+1:]:
                    return False
                new_word_ind = word_list[last_word+1:].index(new_word)+last_word+1
                if new_word_ind < last_word:
                    return False
                last_word = new_word_ind
                new_word_list.append(new_word)
            return True

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

        possible_roots = (self.grammar_by_root[consts.S] +
                          self.grammar_by_root[consts.NP] +
                          self.grammar_by_root[consts.VP])
        horizon[depth].update(entry for entry in possible_roots if consistent(entry) and
                              not entry.tree.adjunct)

        good_subs = [sub_entry for sub_entry in self.grammar_by_type[consts.SUB]
                     if consistent(sub_entry)]
        good_ins = [ins_entry for ins_entry in self.grammar_by_type[consts.INS]
                    if consistent(ins_entry)]

        successes = []

        while depth <= max_depth and len(horizon[depth])>0:
            num_added = [0,0]
            print("{} entries at depth {}".format(len(horizon[depth]), depth))
            horizon[depth+1] = set()
            for i, horizon_entry in enumerate(horizon[depth]):
                if successful(horizon_entry):
                    successes.append(horizon_entry)

                for ins_entry in good_ins:
                    for new_entry in horizon_entry.combine(ins_entry, edge_conditionals):
                        if consistent(new_entry):
                            lex = [x.symbol for _,x in new_entry.lexical]
                            num_added[0]+=1
                            horizon[depth+1].add(new_entry)

                for sub_entry in good_subs:
                    for new_entry in horizon_entry.combine(sub_entry, edge_conditionals):
                        if consistent(new_entry):
                            lex = [x.symbol for _,x in new_entry.lexical]
                            num_added[1]+=1
                            horizon[depth+1].add(new_entry)
            #print("here's a sample:")
            #print("we had added {} trees from inserts and {} trees from subs".format(
            #        *tuple(num_added)))
            #for anexample in list(horizon[depth+1])[:30]:
            #    lex = [x.symbol for _,x in anexample.lexical]
                #cprint("entry: {}".format(anexample),'f')
                #cprint("lex: {}".format(lex), 'f')
                #cprint("points(adjoin,sub):",'w')
                #for point in anexample.adjoin_points:
                #    cprint("\t adjoin: {}".format(point),'w')
                #for point in anexample.subst_points:
                #    cprint("\t subst: {}".format(point),'w')

            depth += 1
        print("Found {} successes".format(len(successes)))
        return [simple_hlf.from_addressbook(success.addressbook) for success in successes]
        #for success in successes:
        #    hlflogic = simple_hlf.from_addressbook(success.addressbook)
        #    yield hlflogic
            #cprint('{:^30}'.format("-"*20), 'f')
            #print(success.tree.verbose_string())
            #print(success.tree.save_str())
            #print(hlflogic)
            #c=lambda x,i:cformat("{}".format(x),i)
            #for key, values in hlflogic.items():
            #    print(cformat("Key",'f')+ " entry: " +
            #           "Head={}, symbol={}".format(c(key.head,'w'), c(key.symbol,'w')))
            #    for value in values:
            #        print(cformat("Value",'0')+" entry: " +
            #              "Head={}, symbol={}".format(c(value.head,'w'), c(value.symbol,'w')))
            #print(simple_hlf.hlf_format(hlflogic))
        #print("Dumping to a file for playing around")
        #with open("testing.pkl", "wb") as fp:
        #    pickle.dump(successes, fp)


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

def test2():
    """
        Starting smaller because this is getting to be too much
    """
    min_grammar = ['(NP* (DT a))',
                 '(NP* (NN motorcycle))',
                 '(NP (NN ride))',
                 '(VP (VB take) (NP))',
                 '(*ADJP (S (VP (TO to) (VP))))',
                 '(*VP (ADJP (JJ ready)))',
                 '(*NP (VP (VBG getting)))',
                 '(*NP (. .))',
                 '(NP* (CD Two))',
                 '(NP (NP (NNS men)))']
    min_grammar = [trees.Entry.make(bracketed_string=elemtree, correct_root=True)
                   for elemtree in min_grammar]
    nlg = generator(min_grammar)
    nlg.set_search("Two men getting ready to take a motorcycle ride .".split(" "))

def test3():
    #shell_logs("hlfdebug")
    min_grammar =   ['(NP* (CD Two))',
                     '(NP (NNS men))',
                     '(*S (. .))',
                     '(NP* (PRP$ their))',
                     '(NP* (JJ red))',
                     '(NP (NNS motorcycles))',
                     '(NP* (DT a))',
                     '(NP (NN church))',
                     '(S (NP) (VP (VB park) (NP)))']
    min_grammar = [trees.Entry.make(bracketed_string=elemtree, correct_root=True)
                   for elemtree in min_grammar]
    nlg = generator(min_grammar)
    nlg.set_search("Two men park their red motorcycles near a church .".split(" "))


if __name__ == "__main__":
    test2()


