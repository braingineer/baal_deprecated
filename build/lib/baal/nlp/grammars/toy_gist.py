from baal.utils.data_structures import trees
import logging, sys

def clean(x):
    return [y.strip() for y in x.split("\n") if len(y.strip())>0]

def make(ainame):
    rules = {'data':data_rules,
             'agentsmith': agentsmith_rules,
             'hal': hal_rules,
             'marvin': marvin_rules}[ainame]
    return [trees.Tree.instantiate(bracketed_string=x)
                   for x in clean(rules)]

hal_rules = """
            (NP (PRP I))
            (S (NP) (VP (V am) (ADJP)))
            (ADJP (JJ sorry))
            (VP (VP) (, ,) (NP))
            (NP (NNP Dave))
            """

agentsmith_rules = """
                    (VP (ADVP never) (VP))
                    (S (VP (V send) (NP)))
                    (NP (NP) (SBAR))
                    (NP (DT) (NN human))
                    (DT a)
                    (SBAR (S))
                    (S (VP (TO to) (VP (V do) (NP))))
                    (NP (NP) (NN))
                    (NP (DT) (NN machine))
                    (NP (NP) (POS 's))
                    (NN job)
                  """

data_rules = """
             (S (S*) (. ?))
             (S (S*) (, ,))
             (S (IN if) (S*))
             (NP (PRP you))
             (S (NP) (VP (V prick) (NP)))
             (NP (PRP me))
             (SQ (VBP do) (NP) (VP))
             (SQ (SQ*) (RB not))
             (NP (PRP I))
             (VP (V leak))
             """

marvin_rules = """
                (S (S) (: :) (S))
                (NP (PRP I))
                (S (NP) (VP (V did) (VP)))
                (VP (VP) (RB n't))
                (VP (V ask) (S))
                (S (VP (TO to) (VP (V be) (VP))))
                (VP (V made))
                (DT no)
                (NP (DT) (CD one))
                (VP (V consulted) (NP))
                (NP (PRP me))
                (S (NP) (VP (VP) (CC or) (VP)))
                (VP (V considered) (NP))
                (NP (PRP my) (NP))
                (NP (NN feelings))
                (VP (VP) (PP (IN in) (NP)))
                (NP (DT) (NN matter))
                (DT the)
                """

def _apply(root,forest,depth,sharedseen):
    if depth > len(forest)*2:
        print "rock bottom"
        return
    print "the root: %s" % root
    for tree in forest:
        for newtree in root.combine(tree):
            print "yielded string: %s" % newtree.yielded
            if newtree.yielded in sharedseen:
                continue
            sharedseen.add(newtree.yielded)
            print "New tree: %s" % newtree
            _apply(newtree,forest,depth+1,sharedseen)

def apply(forest):
    root_symbols = ["S"]
    for tree in forest:
        if tree.root.symbol in root_symbols and \
            all([False if isinstance(child,trees.FootNode) else True
                 for child in tree.root.children]):
            print "using %s" % tree.root
            _apply(tree,forest,0,set())
    # root = trees.Tree.instantiate(bracketed_string='(ROOT (S))')
    # _apply(root,forest,0,set())



