"""
    Bayesian Hiercharical Clustering / Grouping

    original @author: vicky froyen
    port @author: brian mcmahan

    This method is directly lifted from
        Heller's paper:
        http://www.gatsby.ucl.ac.uk/~heller/bhcnew.pdf
        and Froyen's PhD Thesis:
        https://rucore.libraries.rutgers.edu/rutgers-lib/42386/

    Bayesian Hierarchical Clustering is a way of performing agglomerative
        clustering such that each cluster is built up by merging two leaf nodes
        when the probability of them being in one cluster is larger than
        the probability of them being in separate clusters and that probability
        is the current best possible merge.

    This means that we check possible merges at every time step.

    For each possible merge, there is two hypotheses:
        Hypotheses 1: Merge the two subclusters into one cluster
        Hypotheses 2: Keep the two subclusters as separate sub clusters

    Procedure:
        Initialization.
            1. Each datapoint is its own cluster. Instantiate parameters for this.
                Specify the parameters at a later point here.
        Iteration over merges.
            1. For each possible merge, check its rank
            2. Pick best merge, create new merge node information
            3. Put it into the stack
"""

import functools
from Queue import PriorityQueue
from scipy.special import gammaln
from scipy.misc import logsumexp
import numpy as np
from math import log

def clever_logsumexp(some_array):
    some_array = np.array(some_array)
    the_trick = some_array.max()
    return the_trick+logsumexp(some_array-the_trick)

def memoized(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer

class Bookkeeping(object):
    __slots__ = ['log_probability', 'log_p_merge', 'log_p_notmerge',
                 'log_p_merged_subtree', 'log_p_merge_marginal',
                 'merge_score', 'log_d_k']
    def __init__(self):
        pass

class BHCAgenda(object):
    """
        Maintain a priority queue for possible merges
    """
    def __init__(self, sufficient_condition=None):
        if not sufficient_condition:
            self.sufficient_condition = lambda score: True
        else:
            self.sufficient_condition = sufficient_condition
        self.priority_queue = PriorityQueue()
        self.graveyard = set()

    def add_many(self, iterable):
        for item in iterable:
            pass

    def pop(self):
        score, (left_index, right_index) = self.priority_queue.get()
        # print "popped with score %2.5f" % score
        # either left or right index is in graveyard, then draw  another
        # or score is not good enough
        # but no matter what, the queue size must be larger than 0
        while ((left_index in self.graveyard or
               right_index in self.graveyard or
               not self.sufficient_condition(score)) and
               self.priority_queue.qsize() > 0):
            # print 'trying a second time. '
            score, (left_index, right_index) = self.priority_queue.get()
            # print 'score this time: %2.5f' % score
        # avoiding queue errors
        if (   (left_index in self.graveyard or
               right_index in self.graveyard or
               not self.sufficient_condition(score)) and
               self.priority_queue.qsize() == 0):
            return  0,0,0

        # bookkeep seen things
        self.graveyard.add(left_index)
        self.graveyard.add(right_index)

        return score, left_index, right_index

    def clear(self):
        self.priority_queue = PriorityQueue()
        self._put = self.strategy(self.priority_queue)
        self.graveyard = set()
        self.manifest = set()

    def push(self, score, left, right):
        """
            The queue needs to have the score as the first item in the tuple
            This is how it orders the queue
        """
        self.priority_queue.put((score, (left, right)))

    def __len__(self):
        return self.priority_queue.qsize()


class BHCSubtree(object):
    """
        Represent each node in the BHC here
    """
    def __init__(self, data, hyper_parameters, index,
                 probability_function, initialize_probabilities=False):
        """
            Note: Data here is treated as an array of values
                  But to make it more managable, it would make sense to have it be
                  an index into a global data state; perhaps the vocabulary
        """
        # print data
        self.data = data
        self.prior = hyper_parameters['alpha']
        self.hyper_parameters = hyper_parameters
        self.logprior = log(self.prior)
        self.d_k = self.logprior # note, see heller's BHC paper; Figure 3
        self.bookkeeper = {}
        self.index = index
        self.probability_function = probability_function
        self.children = []

        if initialize_probabilities:
            self.log_probability = self.probability_function(self.data)

    def initmerge(self, other):
        """
        initialize probability of merging
            and storing things for O(1) lookup later
        """

        # stolen from Heller's Figure 3 formula. Variable names are bad.
        # alphagam = log ( alpha * gamma(n_k)) where alpha is hyper paramater
        #                and the n_k is the datapoints in the merge
        # combined_dks = log(d_leftk * d_rightk) where d_left/rightk is the d_k
        #                   from the merging subtrees

        alphagam = self.logprior+gammaln(len(self)+len(other))
        combined_dks = self.d_k+other.d_k
        log_d_k = clever_logsumexp([alphagam, combined_dks])
        log_pi_k = alphagam - log_d_k

        """
        note: fill this in later
               considerations:
                    1. keep track of this as we do merges because it's expensive
                    2. I don't think I want to create the new object here
                        so we have to do it cheaply
                    3. (2) might mean we treat the self likelihood function diff
        """
        """
        make_set = lambda l: set([x for x,y in l])
        thispos = [(x,y) for x,y in self.data if y]
        thisposset = make_set(thispos)
        otherpos = [(x,y) for x,y in other.data if y]
        otherposset = make_set(otherpos)

        thisneg = [(x,y) for x,y in self.data if not y and x not in otherposset]
        otherneg = [(x,y) for x,y in other.data if not y and x not in thisposset]
        """
        new_pf = self.probability_function.make()
        new_vals = self.data.copy()
        new_vals.update(other.data)
        log_probability = new_pf(new_vals)

        log_p_merge = log_pi_k + log_probability
        log_p_notmerge = ((combined_dks - log_d_k) +
                      self.log_probability + other.log_probability)

        log_p_merged_subtree = clever_logsumexp([log_p_merge, log_p_notmerge])

        merge_score = log_p_merge - log_p_notmerge

        log_p_merge_marginal = log_p_merge - log_p_merged_subtree
        #    __slots__ = ['log_probability', 'log_p_merge', 'log_p_notmerge',
        #         'log_p_merged_subtree', 'log_p_merge_marginal', 'merge_score']

        new_book = Bookkeeping()
        new_book.log_probability = log_probability
        new_book.log_p_merge = log_p_merge
        new_book.log_p_notmerge = log_p_notmerge
        new_book.log_d_k = log_d_k
        new_book.log_p_merged_subtree = log_p_merged_subtree
        new_book.log_p_merge_marginal = log_p_merge_marginal
        new_book.merge_score = merge_score

        self.bookkeeper[other.index] = new_book

        return merge_score

    def finalmerge(self, other, index):
        """
        look up the merge information initialized earlier
        return it here

        Important! This will return a new tree.  this new true should have
        a reference to self and other, or a copied reference
        The reason for this is because it's a recursive data structure at the moment
        if you don't want to do this and you want to keep a flat top level structure
        then keep the total tree list in the run method

        Note: the current initmerge,finalmerge pair might be inefficient
        Note(cont): however, naive first, optimize later
        """
        new_vals = self.data.copy()
        new_vals.update(other.data)
        newt = BHCSubtree(new_vals, self.hyper_parameters,
                          index, self.probability_function.make(), True)
        this_book = self.bookkeeper[other.index]
        newt.d_k = this_book.log_d_k
        newt.merge_score = this_book.merge_score
        newt.log_p_merge_marginal = this_book.log_p_merge_marginal
        newt.log_probability = this_book.log_probability
        newt.log_p_merged_subtree = this_book.log_p_merged_subtree
        newt.children.extend([self, other])

        return newt

    def __len__(self):
        return len(self.data)

class BayesianHierarchicalClustering(object):
    def __init__(self, hyper_parameters, probability_class,
                 sufficient_condition=None):
        """
            Initialize the BHC algorithm

            Use this for clustering things with a dynamic programming style
                dirichlet computation.  See Heller's BHC paper (insert link)

            Args:
                hyper_parameters: a dictionary that specifies hyper parameters
                                  currently accepts 'alpha' for the dirichlet
                probability_class: a factory class with a class method 'make'
                                   the class should be callable and take as input
                                   data and output either probability or log probability
                sufficient_condition: a function which takes as input the merge_score
                                      and outputs True or False.  Can be used to
                                      only accept certain items off the agenda
                                      say, for example:
                                        items which have a probability of
                                        merging larger than probability of
                                        not merging (since merge_score is in
                                                     log space: merge_score > 0)
        """
        self.sufficient_condition = sufficient_condition
        self.hyper_parameters = hyper_parameters
        self.probability_class = probability_class

    def set_alpha(self, alpha):
        self.hyper_parameters['alpha'] = alpha

    def run(self, data):
        """
            The running procedure

            Note: flippantly call things left and right
                  this has nothign to do with tree composition because
                  the dirichlet prior allows for swappability
                  this is merely the order in which the data structs expect things
                  the merge info is always stored in the left
        """

        self.agenda = BHCAgenda(self.sufficient_condition)
        subtrees = self.initialize(data)

        # print subtrees
        cur_index = len(subtrees)
        #Continue until we have nothing left to run on
        while len(self.agenda) > 0:
            # get the score of the merge and the indices
            score, left_i, right_i = self.agenda.pop()
            # print "popped %s, %d, %d" % (score, left_i, right_i)

            #special condition of queue running out
            if not score and not left_i and not right_i:
                break
                #return subtrees

            # remove the indicies from the list
            left_tree = subtrees.pop(left_i)
            right_tree = subtrees.pop(right_i)

            # update indices for book keeping
            new_index, cur_index = cur_index, cur_index + 1

            # do the deed
            new_tree = left_tree.finalmerge(right_tree, new_index)

            # propagate the new possible merges
            for j, tree_j, in subtrees.items():
                self.agenda.push(new_tree.initmerge(tree_j), new_index, j)

            # add it to the subtree list
            subtrees[new_index] = new_tree
        # end while

        #print "Finished"
        #print "%d subtrees" % len(subtrees)
        #for st_i, subtree in subtrees.items()[::-1]:
        #    print "Subtree; id #:%d" % st_i
        #    print "Data: %s" % subtree.data
        #    #print "Merge score: %s" % subtree.merge_score
        #    print "Posterior: %s" % subtree.probability_function.posterior[:10]

        return subtrees


    def initialize(self, data):
        new_p = lambda: self.probability_class.make
        leaves = {d_i:BHCSubtree(datum, self.hyper_parameters,
                                 d_i, new_p()(), True)
                      for d_i, datum in enumerate(data)}
        available = {}
        for i, leaf_i in leaves.items():
            for j, leaf_j in leaves.items()[i+1:]:
                self.agenda.push(leaf_i.initmerge(leaf_j), i, j)
        return leaves
