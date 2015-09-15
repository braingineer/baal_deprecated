"""
Tree form of the grammar in Jurafsky Martin's Speech & Language Processing
                                        (page 428, Figure 13.1)
@author bcmcmahan

Notes:
       (5/15/2015)
    - They include this weird [Nominal -> Nominal Noun] that I don't see
        the point of.  They don't give any example where it's needed.
        Though, I guess it does all having a noun on the right side of a
        prepositional phrase: Nominal -> (Nominal PP) Noun
        But then, the RHS Nominal will either resolve to Noun or it will keep
        embedding more nouns and PPs to the left. Let's say it does that. And
        alternates. What would be the equivalent rule?
        Maybe a PP insert with a NP substitution site on the left.
        The only problem.. I currently mark recursion like that as insertion.
        crap. Well. I wont' allow such craziness for now. Ask Matthew tomorrow.

        Updated 5/15/2015
    -  So I fixed the insertion point problem. Add asterisks to foot nodes to
        mark for insertion.

"""
from baal.utils.data_structures import trees
from random import shuffle

def clean(x):
    return [y.strip() for y in x.split("\n") if len(y.strip())>0]

def make():
    return [trees.Tree.instantiate(bracketed_string=x)
                   for x in clean(L1_rules)]

L1_rules = """
            (NP (PRP I))
            (NP (PRP she))
            (NP (PRP me))
            (NP (NNP Houstan))
            (NP (NNP NWA))
            (NP (NN book))
            (NP (NN flight))
            (NP (NN meal))
            (NP (NN money))

            (NN book)
            (NN flight)
            (NN meal)
            (NN money)

            (NP (DT that) (NN))
            (NP (DT this) (NN))
            (NP (DT a) (NN))
            (NP (DT the) (NN))

            (S (NP) (VP (V book) (NP)))
            (S (VP (V book) (NP)))
            (S (NP) (VP (V include) (NP)))
            (S (VP (V include) (NP)))
            (S (NP) (VP (V prefer) (NP)))
            (S (VP (V prefer) (NP)))

            (NP (NP*) (PP (IN from) (NP)))
            (NP (NP*) (PP (IN to) (NP)))
            (NP (NP*) (PP (IN on) (NP)))
            (NP (NP*) (PP (IN near) (NP)))
            (NP (NP*) (PP (IN through) (NP)))

            (S (Aux does) (S*))
            """

def tests():
    forest = make()
    for tree in forest:
        print tree

    book = forest[13]
    that = forest[9]
    flight = forest[6]

    print book, that, flight

    print "frontier: "
    print flight.frontier[0]

    print "book that flight:"
    for newt1 in flight.combine(that):
        print newt1
        for newt2 in book.combine(newt1):
            print newt2


    I = forest[0]
    prefer = forest[16]
    money = [8]

    # apply(forest)

def _apply(root,forest,depth,sharedseen):
    if depth > 6:
        print "rock bottom"
        return
    shuffle(forest)
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

if __name__=="__main__":
    tests()
