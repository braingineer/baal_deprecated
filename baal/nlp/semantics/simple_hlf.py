from baal.utils.hlf import gensym, unboundgensym
from baal.utils.general import nonstr_join
from collections import deque

def from_addressbook(addressbook):
    """
        Take a tree and make it into logical form

        Observations motivating procedure:
            1. The parent's head word is the function over its arguments
            2. Its arguments are marked as complements
            3. Its spine index marks itself
            4. Things that aren't its spine and aren't its complements are adjuncts
            5. Adjuncts are functions on the headword.

        Procedure:
            Iterate through address, tree pairs
                1. If address is longer, we have descended into children
                2. if parent's spine index is this address's last index, it's on spine
                3. if this child has a marker "complement", it's an argumetn of parent
                4. If the last two aren't true, it's an adjunct.
            Note: we do nothing for spines, but continue iterating (presumably, the head word won't change)
    """

    enter_child_cond = lambda addr, last_addr: len(addr) > len(last_addr)
    # exit the set of children of a node
    exit_child_cond = lambda addr, last_addr: len(addr) < len(last_addr)

    logical_form = dict()
    headbook = dict()

    stack = deque()

    parent_address, parent = addressbook[0]
    last_address, last_tree = addressbook[1]

    for address, tree in addressbook[1:]:
        # print address, tree


        if enter_child_cond(address, last_address):
            # this means we are decending a level.
            # print "child condition."
            # print "pushing parent %s" % parent.head
            # print "new parent %s" % last_tree.head
            # print "current tree %s" % tree.head
            stack.append((parent, parent_address))
            parent, parent_address = last_tree, last_address


        elif exit_child_cond(address, last_address):
            # this means we are moving up a level
            # print "exiting child condition"
            # print "throwing %s away" % parent.head
            while len(parent_address) >= len(address):
                parent, parent_address = stack.pop()
                # print address, parent_address

            # print "popped %s as new (old) head" %  parent.head
            # print "current tree: %s" % tree.head

        else:
            # this means we are at the same level as last iteration
            pass


        psym = gensym(head=parent.head, address=parent_address, symbol=parent.symbol)
        csym = (gensym(head=tree.head, address=address, symbol=tree.symbol)
                if len(tree.head) > 0 else
                gensym(head=tree.symbol, address=address, symbol=tree.symbol))
        if parent.spine_index == address[-1]:
            # print "spine condition."
            # print "parent: %s<%s>" % (parent.symbol, parent.head)
            # print "current: %s<%s>" % (tree.symbol, tree.head)
            pass
        elif tree.complement:
            # print "complement condition"
            # print "parent: %s<%s>" % (parent.symbol, parent.head)
            # print "current: %s<%s>" % (tree.symbol, tree.head)
            # print "psym %s for %s" % (psym, parent.head)
            # print "csym %s for %s" % (csym, tree.head)
            if len(tree.children) == 0:
                csym = unboundgensym(head=tree.symbol,address=address, symbol=tree.symbol)
                logical_form.setdefault(psym, [psym]).append(csym)
            else:
                logical_form.setdefault(psym, [psym]).append(csym)
                logical_form.setdefault(csym, [csym])
        else:
            # print "Adjunct Condition"
            # print "parent: %s<%s>" % (parent.symbol, parent.head)
            # print "current: %s<%s>" % (tree.symbol, tree.head)
            # print tree.complement
            # print tree.adjunct
            # print parent.spine_index
            # print "adding psym %s to csym %s for head %s" % (psym, csym, tree.head)
            logical_form.setdefault(csym, [csym]).append(psym)
        # print "--\n--\n"
        last_address, last_tree = address, tree

    hlf_make = lambda func,variables: "%s(%s)" % \
                                      (func.head, nonstr_join(variables, ','))
    hlf_items = sorted(logical_form.items(), key = lambda x: str(x[1][0]))
    # print hlf_items
    hlf_expr = " & ".join([hlf_make(f,vs) for f,vs in hlf_items])

    # print hlf_expr

    return logical_form

def find_dominant(logical_form):
    """ this is a bad algorithm. fix later. it's basically a bubble sort """
    sorted = {k:i for i,k in enumerate(logical_form.keys())}
    for sym1, args1 in logical_form.items():
        for sym2, args2 in logical_form.items():
            if sym2 in args1 and sorted[sym2]>sorted[sym1]:
                t = sorted[sym1]
                sorted[sym1] = sorted[sym2]
                sorted[sym2] = t
            elif sym1 in args2 and sorted[sym1]>sorted[sym2]:
                t = sorted[sym1]
                sorted[sym1] = sorted[sym2]
                sorted[sym2] = t
    return sorted

def post_process_hlf(logical_form):
    post_processed = {}
    dominant = None
    sorted = find_dominant(logical_form)
    # print 'blah'
    for sym, args in logical_form.items():
        if not any([(sym in v and k.pos_symbol == sym.pos_symbol) for k,v in post_processed.items()]):
            post_processed[sym] = args
        else:
            # print 'here'
            kvs = post_processed.items()
            for k,v in kvs:
                if sym in v and k.pos_symbol == sym.pos_symbol:
                    # print 'del'
                    del post_processed[k]
                    post_processed[sym] = args

    return post_processed


def hlf_format(logical_form):
    hlf_make = lambda func,variables: "%s(%s)" % (func.head, nonstr_join(variables, ','))
    hlf_items = sorted(logical_form.items(), key = lambda x: str(x[1][0]))
    hlf_expr = " & ".join([hlf_make(f,vs) for f,vs in hlf_items])
    return hlf_expr
