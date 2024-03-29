"""
Finding Dependencies and Heads in Derivation Trees

- For 1-ply subtrees, we have a root and a set of children.
- In tree-annotated corpora, it's not indicated:
    - which of the children are the head,
    - which of the children are the complements
    - which of the children are the adjuncts

[Wagerman95] used a head percolation table to propogate heads upward
from the leaves to the root.  He was interested in this information for a
PCFG

[Collins1999] used these tables in his dissertation to further head-driven
PCFG parsing.  Collins extended this to finding the constituents of the heads
so that they can be distinguished from the adjuncts.

Required datastructure (see data_structures for wrappers):
    tree(object):
        head_word = ""
        head_index = -1
        adjuncts = []
        substitutions = []


@author bcmcmahan
"""
import baal
from baal.utils import AlgorithmicException
from baal.utils.general import backward_enumeration, flatten, SimpleProgress, cformat
from collections import defaultdict
import argparse
from functools import wraps
import logging
try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger("treecut")

def debugrule(func):
    @wraps(func)
    def debug_and_call(*args,**kwargs):
        if len(args)==2:
            children, headlist = args
        else:
            return func(*args, **kwargs)
        fname = cformat(func.func_name, 'w')
        cname = cformat("{}".format(children), 'f')
        logger.debug("inside {} search with children: {} and headlist: {}".format(fname,
                                                                           cname,
                                                                           headlist))
        ind, suc = func(*args, **kwargs)
        csel = cformat("{}".format(children[ind]),'f')
        logger.debug("selecting {} at index {}; successful={}".format(csel, ind, suc))
        return ind, suc
    return debug_and_call

class CollinsMethod(object):
    headruleset = {
            "ROOT": [("right",)],
            "ADJP": [("left", "NNS", "QP", "NN", "$", "ADVP", "JJ",
                      "VBN", "VBG", "ADJP", "JJR", "NP", "JJS", "DT",
                      "FW", "RBR", "RBS", "SBAR", "RB")],
            "ADVP": [("right", "RB", "RBR", "RBS", "FW", "ADVP",
                       "TO", "CD", "JJR", "JJ", "IN", "NP", "JJS", "NN")],
            "CONJP": [("right", "CC", "RB", "IN")],
            "FRAG": [("right",)],
            "INTJ": [("left",)],
            "LST": [("right", "LS", ":")],
            "NAC": [("left", "NN", "NNS", "NNP", "NNPS", "NP", "NAC",
                      "EX", "$", "CD", "QP", "PRP", "VBG", "JJ", "JJS",
                      "JJR", "ADJP", "FW")],
            "NX": [("left",)],
            "PP": [("right", "IN", "TO", "VBG", "VBN", "RP", "FW")],
            "PRN": [("left",)],
            "PRT": [("right", "RP")],
            "QP": [("left", "$", "IN", "NNS", "NN", "JJ", "RB", "DT",
                    "CD", "NCD", "QP", "JJR", "JJS")],
            "RRC": [("right", "VP", "NP", "ADVP", "ADJP", "PP")],
            "S": [("left", "TO", "IN", "VP", "S", "SBAR",
                   "ADJP", "UCP", "NP")],
            "SBAR": [("left", "WHNP", "WHPP", "WHADVP", "WHADJP", "IN",
                      "DT", "S", "SQ", "SINV", "SBAR", "FRAG")],
            "SBARQ": [("left", "SQ", "S", "SINV", "SBARQ", "FRAG")],
            "SINV": [("left", "VBZ", "VBD", "VBP", "VB", "MD", "VP",
                      "S", "SINV", "ADJP", "NP")],
            "SQ": [("left", "VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ")],
            "UCP": [("right",)],
            "VP": [("left", "TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG",
                    "VBP", "AUX", "AUXG", "VP", "ADJP", "NN", "NNS", "NP")],
            "WHADJP": [("left", "CC", "WRB", "JJ", "ADJP")],
            "WHADVP": [("right", "CC", "WRB")],
            "WHNP": [("left", "WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP")],
            "WHPP": [("right", "IN", "TO", "FW")],
            "X": [("right",)],
            "NP": [("rightdis", "NN", "NNP", "NNPS", "NNS",
                    "NX", "POS", "JJR"),
                   ("left", "NP"),
                   ("rightdis", "$", "ADJP", "PRN"),
                   ("right", "CD"),
                   ("rightdis", "JJ", "JJS", "RB", "QP")],
            "TYPO": [("left",)],
            "EDITED": [("left",)],
            "XS": [("right", "IN")],
            "": [('left',)]
                    }


    class CMWrapper:
        @staticmethod
        @debugrule
        def left(children, headlist):
            """ head symbol preference then forward order of children preference """
            for head_symbol in headlist:
                for c_i, child in enumerate(children):
                    if child.symbol == head_symbol:
                        return c_i, True
            return 0, False


        @staticmethod
        @debugrule
        def right(children, headlist):
            """ head symbol preference then backward order of children preference """
            for head_symbol in headlist:
                for c_i, child in backward_enumeration(children):
                    if child.symbol == head_symbol:
                        return c_i, True
            return -1, False

        @staticmethod
        @debugrule
        def leftdis(children, headlist):
            """ forward order of children preference """
            for c_i, child in children:
                if child.symbol in headlist:
                    return c_i, True
            return 0, False

        @staticmethod
        @debugrule
        def rightdis(children, headlist):
            """ backward order of children preference """
            for c_i, child in backward_enumeration(children):
                if child.symbol in headlist:
                    return c_i, True
            return -1, False

        @staticmethod
        @debugrule
        def leftexcept(children, headlist):
            """ excluding symbols, forward order of children preference """
            for c_i, child in enumerate(children):
                if child.symbol not in headlist:
                    return c_i, True
            return 0, False

        @staticmethod
        @debugrule
        def rightexcept(children, headlist):
            """ excluding symbols, backward order of children preference """
            for c_i, child in backward_enumeration(children):
                if child.symbol not in headlist:
                    return c_i, True
            return -1, False

    headrule_functions = {"left": CMWrapper.left,
                          "right": CMWrapper.right,
                          "leftdis": CMWrapper.leftdis,
                          "rightdis": CMWrapper.rightdis,
                          "leftexcept": CMWrapper.leftexcept,
                          "rightexcept": CMWrapper.rightexcept}

    complementruleset = {"first_condition":
                         [(set(("NP", "SBAR", "S")), set(["S"])),
                          (set(("NP", "SBAR", "S", "VP")), set(["VP"])),
                          (set(("NP", "S")), set(["SBAR"]))],
                         "second_condition":
                         set(("ADV", "VOC", "BNF", "DIR", "EXT", "LOC", "MNR",
                              "TMP", "CLR", "PRP"))
                         }


    @staticmethod
    def mark_complements(parent, children):
        for c_i, child in enumerate(children):
            if parent.symbol == "PP" and c_i == 1:
                child.complement = True
            elif c_i == parent.head_index:
                continue
            else:
                child.complement = CollinsMethod.is_complement(parent, child)
        return children

    @staticmethod
    def is_complement(parent, child):
        first_cond = CollinsMethod.complementruleset["first_condition"]

        first_condition_bool = []
        for nt_set, pnt_set in first_cond:
            if child.symbol in nt_set and parent.symbol in pnt_set:
                first_condition_bool.append(True)
            else:
                first_condition_bool.append(False)
        first_condition_bool = any(first_condition_bool)

        second_cond = CollinsMethod.complementruleset["second_condition"]
        return first_condition_bool and child.semantictag not in second_cond


def populate_annotations(tree):
    """
        Input: a tree with subtree children
        Output: each tree object is annotated with head, adjunct, or substitution
    """
    parent = tree
    children = tree.children
    parent = _annotate(parent, children)
    parent, children = select_head(parent, children)
    return parent

def _annotate(parent, children):
    if parent.lexical:
        parent.head = parent.symbol
        return parent


    if len(children)==0:
        # This is the lexical node.
        # Shouldn't be here though...
        # print parent, children
        children[0].head = children[0].symbol
        raise ValueError, ("Shouldn't be here. tree_enrichment, lexical node",
                           "Unless.. Maybe we don't see only pre-terminals before leaves")

    if len(children)==1 and children[0].lexical:
        head = children[0]
        parent.head_index = 0
        parent.head = children[0].symbol

    else:
        # print len(children)
        # print children
        for child in children:
            child = _annotate(child, child.children)

        parent, children = select_head(parent, children)
        # print "with a parent as %s" % parent.symbol
        # print "the head is index %d" % parent.head_index

        adj, subs, children = mark_dependencies(parent, children)
        # print "we found adjuncts: %s" % [a.symbol for a in adj]
        # print "we found substitutions: %s" % [s.symbol for s in subs]
        # print "but overall, we have children: %s" % [(ci, c) for ci, c in enumerate(children)]

        parent.adjuncts, parent.substitutions, parent.children = \
            mark_dependencies(parent, children)

    return parent




def select_head(parent, children):
    """
        Input:
            parent: the parent Non-Terminal
            children: a list of symbols, either Non-Terminal or Terminal

        Procedure:
            Find rule for parent
            Proceed by parameters of rule to select head
            Default if rule matches nothing
            Annotate head
            Return

        Output:
            Returns (parent,children) with head child annotated
    """
    rules = CollinsMethod.headruleset[parent.symbol]
    logger.debug("=======\n starting select head with "+cformat("{}".format(parent), '2'))
    logger.debug("rules: {}".format(rules))
    # print "parent symbol", parent.symbol
    # print "rules",  rules
    for rule in rules:
        # print rule
        search_method, argset = rule[0], rule[1:]
        # print rules
        func = CollinsMethod.headrule_functions[search_method]
        head_ind, success = func(children, argset)
        if success:
            #head_ind = len(children)-1 if head_ind < 0 else head_ind
            head = children[head_ind]
            head.on_spine = True
            parent.head_index = head_ind
            # print head
            parent.head = head.head
            parent.head_symbol = head.symbol
            parent.spine_index = head_ind
            logger.debug("=======")
            return parent, children
    raise AlgorithmicException("We shouldn't be here")

def mark_dependencies(parent, children):
    """
        Input:
            parent: the parent Non-Terminal
            children: a list of symbols, either Non-Terminal or Terminal

        Procedure:
            iterate children
            determine if child meets complement rules
            annotate complement if it does
            annotate adjunct if it does not

        Output:
            returns (parent,children) with

    """
    children = CollinsMethod.mark_complements(parent, children)
    adjuncts = [child for c_i, child in enumerate(children)
                if not child.complement and not parent.head_index == c_i]
    complements = [child for child in children if child.complement]
    #adjuncts, complements = specialcases(children, adjuncts, complements)
    return adjuncts, complements, children

def specialcases(children, adjuncts, complements):
    if (len(adjuncts) > 1 and
        all([adj.symbol==adjuncts[0].symbol for adj in adjuncts])):
        adjuncts = []
    #for adj in adjuncts:
    #    if adj.symbol == ',' and children.index(adj)
    return adjuncts, complements

def annotation_cut(tree):
    """
        Given an annotated tree, return the forest of subtrees which
            represents the annotated split (head with spine, complements via
            substitution, and adjuncts via insertion)
k
    """
    # print "top level annotation cut"
    # print "repr of the tree: %s" % repr(tree)
    #decomposed = []
    #fix_spine(tree)
    #decomposed.extend(flatten(_recursive_cut(t)
    #                          for t in tree.excise_substitutions()))
    #decomposed.extend(flatten(_recursive_cut(t)
    #                          for t in tree.excise_adjuncts()))

    return  _recursive_cut(tree) + [tree]


def _recursive_cut(subtree):
    decomposed = []
    recursed = []
    # print 'substitutions: %s' % subtree.substitutions
    # print 'adjuncts: %s' % subtree.adjuncts
    fix_spine(subtree)
    # print 'substitutions: %s' % subtree.substitutions
    # print 'adjuncts: %s' % subtree.adjuncts
    subs = [annotation_cut(t) for t in subtree.excise_substitutions()]
    #print "SUBS: %s" % subs
    adjs = [annotation_cut(t) for t in subtree.excise_adjuncts()]
    #print "ADJS: %s" % adjs
    decomposed.extend(flatten(subs))
    decomposed.extend(flatten(adjs))
    spine = _get_spine(subtree)
    if spine:
        # print "in spine condition: %s" % spine
        recursed = _recursive_cut(spine)
    #print "finishing %s" % subtree
    #print "DECOMPOSED: %s" % decomposed
    #print "RECURSED: %s" % recursed
    return decomposed + recursed

def _get_spine(subtree):
    if len(subtree.children) > 0:
        return subtree.children[subtree.spine_index]
    else:
        return False

def fix_spine(tree):
    """
        we can't excise our spine.

        The tree should have a spine index. if its spine is in the adjuncts
        or the substitutions, remove it, and update all of the bookkeeping stuff
        inside the fixed spine.
    """
    if len(tree.children) > 0:
        spine = _get_spine(tree)
        # print "tree %s " % tree
        # print "Has a spine: %s" % spine
        if spine in tree.substitutions:
            tree.substitutions.remove(spine)
        if spine in tree.adjuncts:
            tree.adjuncts.remove(spine)




def excise_substitutions(tree):
    pass

def excise_adjuncts(tree):
    pass

def gentest(instr):
    tree, abook = baal.utils.data_structures.trees.from_string(instr)
    # print tree.verbose_string()
    populate_annotations(tree)
    # print 'new 3'
    # return tree
    copied = tree.clone()
    cuts = annotation_cut(tree)
    # print "\n"*3, "--"*10
    print "We have found %d cuts" % len(cuts)
    for i,cut in enumerate(cuts):
        print "Cut #%d: " % i, repr(cut), "\n"

    return cuts, copied

def test():
    a="""
    (S (PP On
         (NP their way))
     (NP-SBJ-1 they)
     (VP stopped
         (PP-LOC at
         (NP (NP every gas station)
             (PP-LOC along
                 (NP the main boulevards))))
     (S-PRP (NP-SBJ *-1)
        (VP to
            (VP question
            (NP the attendants))))))"""
    return gentest(a)

def test2():
    b="""
    ( (S
    (PP (IN On)
      (NP (PRP$ their) (NN way) ))
    (, ,)
    (NP-SBJ-1 (PRP they) )
    (VP (VBD stopped)
      (PP-LOC (IN at)
        (NP
          (NP (DT every) (NN gas) (NN station) )
          (PP-LOC (IN along)
            (NP (DT the) (JJ main) (NNS boulevards) ))))
      (S-PRP
        (NP-SBJ (-NONE- *-1) )
        (VP (TO to)
          (VP (VB question)
            (NP (DT the) (NNS attendants) )))))
    (. .) ))"""
    return gentest(b)


def test3():
    c="""
    ( (S
    (ADVP (RB Finally) )
    (, ,)
    (PP-LOC (IN at)
      (NP
        (NP (NNP Ye) (NNP Olde) (NNP Gasse) (NNP Filling) (NNP Station) )
        (PP-LOC (IN on)
          (NP (NNP Avocado) (NNP Avenue) ))))
    (, ,)
    (NP-SBJ (PRP they) )
    (VP (VBD learned)
      (SBAR (DT that)
        (S
          (NP-SBJ-1 (PRP$ their) (NN man) )
          (, ,)
          (S-ADV
            (NP-SBJ-2 (-NONE- *-1) )
            (VP (VBG having)
              (VP (VBN paused)
                (S-PRP
                  (NP-SBJ (-NONE- *-2) )
                  (VP (TO to)
                    (VP (VB get)
                      (NP
                        (NP (NN oil) )
                        (PP (IN for)
                          (NP (PRP$ his) (NN car) )))))))))
          (, ,)
          (VP (VBD had)
            (VP (VBN asked)
              (PP (IN about)
                (NP
                  (NP (DT the) (NN route) )
                  (PP (TO to)
                    (NP (NNP San) (NNP Diego) )))))))))
    (. .) ))"""
    return gentest(c)

def test4():
    path = "/home/cogniton/research/data/LDC/treebank3/treebank_3/parsed/mrg/brown/cr/cr01.mrg"
    with open(path) as fp:
        lines = fp.readlines()
    #print lines
    start_indices = [i for i,x in enumerate(lines) if x[0] == "("]
    # print start_indices
    instrs = []
    yesno = raw_input("About to parse and print up to %d trees " % len(start_indices) +
                      "(You'll choose a start,end soon). Are you sure? (yes/no): ")
    if yesno.lower() == "no":
        print "I don't blame you."
        yesno = raw_input("Can I show you the nth smallest one? ")
        if yesno.lower() == "yes":
            n = int(raw_input("whats the n: "))
            lines_ind_start, lines_ind_end = sorted(
                [(start_indices[i-1], second_ind)
                 for i,second_ind in enumerate(start_indices[1:],1) if (second_ind-start_indices[i-1])>1],
                key=lambda x:x[1]-x[0])[n]
            liststr = [x for x in lines[lines_ind_start:lines_ind_end] if len(x) > 0]
            print("(start,end) = (%d,%d)" % (lines_ind_start, lines_ind_end))
            print "".join(liststr)
            gentest("".join(liststr))
        print "Quitting"
        return

    start = int(raw_input("Please type a starting index number from [0,%d]: " % (len(start_indices)-1) ))
    end = int(raw_input("Choose an ending index number from [%d,%d]: "
                         % (start+1, len(start_indices)-1) ))

    if end >= len(start_indices):
        print "Bad end number. using %d" % (len(start_indices)-1)
    elif end <= start:
        print "Have to be bigger than start. using %d" % (len(start_indices)-1)

    for i,ind2 in enumerate(start_indices[start+1:end+1], start+1):
        ind1 = start_indices[i-1]
        print "(start,end) = (%d,%d)" % (ind1, ind2)
        liststr = [x for x in lines[ind1:ind2] if len(x) > 0]
        #print liststr
        instrs.append("".join(liststr))
    for instr in instrs:
        print instr
        gentest(instr)


def test5():
    path = "/home/cogniton/research/data/LDC/treebank3/treebank_3/parsed/mrg/brown/cr/cr01.mrg"
    with open(path) as fp:
        lines = fp.readlines()
    #print lines
    start_indices = [i for i,x in enumerate(lines) if x[0] == "("]
    # print start_indices
    instrs = []

    for i,ind2 in enumerate(start_indices[1:], 1):
        ind1 = start_indices[i-1]
        liststr = [x for x in lines[ind1:ind2] if len(x) > 0]
        instrs.append("".join(liststr))

    all_cuts = defaultdict(lambda: 0)

    for instr in instrs:
        tree, abook = baal.utils.data_structures.trees.from_string(instr)
        populate_annotations(tree)
        copied = tree.clone()
        cuts = annotation_cut(tree)
        for i,cut in enumerate(cuts):
            all_cuts[repr(cut)]+=1
    print "found %s cuts" % len(all_cuts.keys())
    print "top cuts: "
    for cut,count in sorted(all_cuts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print "\t{} :: {}".format(cut,count)

def get_trees(infile):
    fix = lambda y: [x.replace("\n","").replace("ROOT","").strip() for x in y]
    with open(infile) as fp:
        capacitor = []
        is_consuming = True
        for i, line in enumerate(fp):

            # catching shitty noise
            if line[0]!="(" and not capacitor:
                continue

            if line[0]=="(" and capacitor:
                yield "".join(fix(capacitor))
                capacitor = []

            capacitor.append(line)



def split_file(infile, outfile,data_n):
    if '.' not in outfile:
        outfile += '.pkl'

    all_cuts = defaultdict(lambda: 0)

    if data_n:
        prog = SimpleProgress(data_n)
        prog.start_progress()

    for instr in get_trees(infile):

        try:
            tree, abook = baal.utils.data_structures.trees.from_string(instr)
        except TypeError as e:
            print "parsing a tree broke"
            print instr
            raise TypeError

        populate_annotations(tree)
        copied = tree.clone()
        cuts = annotation_cut(tree)

        for i,cut in enumerate(cuts):
            all_cuts[cut.save_str()]+=1

        # update us on timing
        if  data_n:
            prog.incr()
            if prog.should_output():
                print prog.update()
                print "TOP 10"
                for cut,count in sorted(all_cuts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print "\t{} :: {}".format(cut,count)

    all_cuts = {k:v for k,v in all_cuts.items()}
    print 'finished. saving now.'
    with open(outfile, 'wb') as fp_out:
        pickle.dump(all_cuts, fp_out)
    print "found %s cuts" % len(all_cuts.keys())
    print "specify top n to show"
    n = int(raw_input("n?: "))
    print "top {} cuts: ".format(n)
    for cut,count in sorted(all_cuts.items(), key=lambda x: x[1], reverse=True)[:n]:
        print "\t{} :: {}".format(cut,count)

def parse_args():
    parser = argparse.ArgumentParser(description='Collins Head Rule on Phrase Structures')
    parser.add_argument('treefile', type=str, help='a file with the to-split trees')
    parser.add_argument('outfile', type=str, help='the filename for the output pickle')
    parser.add_argument('-n', '--num_trees', type=int, default=0,
                        help='the number of trees in the file if you know it')
    return parser.parse_args()

if __name__ == "__main__":
    # test()
    # test2()
    # test3()
    # test4()
    # test5()
    args = parse_args()
    split_file(args.treefile, args.outfile, args.num_trees)
