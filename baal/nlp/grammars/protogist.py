"""
protyping for gist (grounded interactive semantics with trees)
"""
from ..corpora.captions import get_all_treestrings
from baal.utils.data_structures import trees

def _makegrammar(treestrings):
    ret = []
    for treestring in treestrings:
        if "RB-" in treestring: continue
        try:
            ret.append(trees.Entry.make(bracketed_string=treestring, correct_root=True))
        except (IndexError, AssertionError) as e:
            print("catching at grammar construction level")
            print("the offender: "+treestring)
    return ret

    # taking this out for now so that we can verbosely check the strings
    #return [trees.Entry.make(bracketed_string=treestring, correct_root=True)
    #        for treestring in treestrings]

def make(specific=None):
    trees8k, trees30k, treescoco = get_all_treestrings()
    if specific:
        grammar_lookup = {"8k":trees8k, "30k":trees30k, "coco":treescoco}
        if specific not in grammar_lookup:
            print("{} is not a valid key".format(specific))
            raise IndexError
        print("Making {}".format(specific))
        return _makegrammar(grammar_lookup[specific])
    else:
        print("Making all grammars")
        grammar8k = _makegrammar(trees8k)
        print("Finished flickr8k.  Making flickr30k.")
        grammar30k = _makegrammar(trees30k)
        print("Finished flickr30k.  Making COOC.")
        grammarcoco = _makegrammar(treescoco)
        try:
            with open("allgrammars.pkl","wb") as fp:
                pickle.dump([grammar8k, grammar30k, grammarcoco], fp)
        except:
            print "pickling failed"

        print("{} grammar entries for flickr8k,".format(len(trees8k)),
              "{} grammar entries for flickr30k,".format(len(trees30k)),
              "{} grammar entries for coco".format(len(treescoco)))
        return grammar8k, grammar30k, grammarcoco
