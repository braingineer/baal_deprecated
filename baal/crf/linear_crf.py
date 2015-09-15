"""
Linear Chain Conditional Random Field
Adapted from Tim Vieira (https://github.com/timvieira/crf)
"""

from baal.utils.general import bigram_enum, reversezip, count_time
from baal.utils.data import split_list
from baal.utils.vocabulary import Vocabulary
from baal.utils import vault
from scipy.misc import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from collections import defaultdict
#from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import re
from random import shuffle
import math
import time

def clever_logsumexp(some_array):
    the_trick = some_array.max()
    return the_trick+logsumexp(some_array-the_trick)

class LinearCRF(object):

    def __init__(self, data, label_vocab,
                 feature_vocab, split_proportions=(0.7, 0.15, 0.15), split=True):
        if split:
            self.training, self.dev, self.test = split_list(
                data, split_proportions)
        else:
            self.training, self.dev, self.test = data

        self.label_vocab = label_vocab
        self.feature_vocab = feature_vocab
        self.weights = np.random.random_sample(size=len(feature_vocab))
        self.sigma = 2
        self.regularizer = lambda weights: sum([x**2/(2*self.sigma**2)
                                                for x in weights])
        self.derived_regularizer = lambda weights: weights / (self.sigma**2)
        self.history = []
        self.subset_perc = 0.3

    @classmethod
    def make(cls, raw_data):
        """
        Data comes in the form of [ Chain, Chain,...]
        Chains are in the form of [ (Observed, Hidden), (Observed,Hidden), ... ]

        This will rely heavily on the Chain objects
        The Chain object is preprocessed so that we have vocabularies over labels and features
        We then populate, per chain, a lookup table for features
        In this lookup table, we store things as a numpy int32 type array.
        These types of arrays can index other arrays to return the items to which they index
        For example: [1,2,3,4,5] as a numpy array and indexed by [1,2] would return [2,3]
            for exactness:
                a=np.array([1,2,3,4,5])
                b=np.array([1,2],dtype=np.int32)
                a[b] # outputs [2,3]
        See the chain object for further documentation. It's been augmented to act as a callable container type

        Notation: I use y and yp for the indices of label and prev_label respectively
        """

        data, label_vocab, feature_vocab = LinearCRF.preprocess(raw_data)
        fmap = lambda *args: feature_vocab.map(*args)
        # make all possible feature tables
        for chain in data:
            # Make the "prior" (starting point of chain) features
            for y, label in label_vocab.enum():
                # store the list of feature indices in the look up table
                chain[0, None, y] = np.fromiter(
                    fmap(chain(0, None, label)), dtype=np.int32)
            # Make all possible features, populate feature table
            for t in xrange(1, len(chain)):
                for y, label in label_vocab.enum():
                    for yp, prev_label in label_vocab.enum():
                        # store the list of feature indices in the look up
                        # table
                        chain[t, yp, y] = np.fromiter(
                            fmap(chain(t, prev_label, label)), dtype=np.int32)
            chain.true_features = LinearCRF.path_features(
                chain, label_vocab.map(chain.true_labels))
            chain.seq_map_features()
        return LinearCRF(data, label_vocab, feature_vocab)

    @classmethod
    def from_vault(cls, storage_items):
        crf_info = dict(storage_items)

        data = crf_info['data']
        label_vocab = crf_info['label_vocab']
        feature_vocab = crf_info['feature_vocab']
        weights = crf_info['weights']

        vault_crf = cls(data, label_vocab, feature_vocab, split=False)
        vault_crf.weights = weights

        return vault_crf

    def vault_items(self):
        crf_info = dict()
        crf_info['data'] = (self.training, self.dev, self.test)
        crf_info['label_vocab'] = self.label_vocab
        crf_info['feature_vocab'] = self.feature_vocab
        crf_info['weights'] = self.weights
        return crf_info

    def calc_edge_potentials(self, chain, weights):
        """
        assuming self.weights has a certain value
        """
        # make the prior edge potentials (0,None,y)
        N, K = len(chain), len(self.label_vocab)
        prior_potentials = np.array([weights[chain[0, None, y]].sum()
                                     for y in xrange(K)])
        edge_potentials = np.zeros((N, K, K))
        # t starts at 1 because prior potentiasl are our 0 state.
        for t in xrange(1, N):
            for y in xrange(K):
                for yp in xrange(K):
                    edge_potentials[t, yp, y] = weights[chain[t,yp,y]].sum()
        # print np.sum(edge_potentials[:,:,:], axis=(0,1,2))
        # edge_potentials[0,0,:] = prior_potentials
        return prior_potentials, edge_potentials

    def forward(self, prior_potentials, edge_potentials, N, K):
        """
        The forward algorithm calculates the probability of the data.
        Edge potentials are transition functions between each possible hidden state
        So, we use dynamic programming to calculate the marginalization over
        every possible path
            for hidden state i (hidden state = pos, named entity, etc)
            alpha[0, i] = prior_potential[i]
            all possible ways of getting from t-1 to i in t, do for all i's, repeat
            alpha[t, i] = log ( sum_j ( exp(alpha[t-1,j], edge_potential[t, j, i])))

        Note: this is log alpha. It is kept in log space for tractibility.
        """

        alpha = np.zeros((N,K))
        alpha[0, :] = prior_potentials
        for t in xrange(1,N):
            prior_alpha = alpha[t-1, :]
            for y in xrange(K):
                # See bottom of file for note 3.
                alpha[t,y] = clever_logsumexp(prior_alpha + edge_potentials[t, :, y])
        return alpha

    def backward(self, prior_potentials, edge_potentials, N, K):
        """
            Calculates the probability of the data.
            Input:
                edge_potentials = energy for all t,yp,y triplets
                N = size of chain
                K = size of latent state space

            Note: this is a log beta. We keep it in log space for tractibilty.
        """
        beta = np.zeros((N,K))
        #beta[N-1,:] = np.ones(K)
        for t in reversed(xrange(0, N-1)):
            last_beta = beta[t+1, :]
            for yp in xrange(K):
                beta[t,yp] = clever_logsumexp(last_beta + edge_potentials[t+1,yp,:])
        return beta

    def objective(self, weights, chains, gradient=True, verbose=True):
        """
            Function to calculate the negative log likelihood and derivative
            so that it can be used in the L-BFGS algorithm.

            Input:
                the weights (called lambda or theta elsewhere)
                    = Kx1 Vector; where K = # Features
                data = Labeled sequence observations

            Procedure:
                Iterate over the chains
                aggregate likelihood terms
                compute the parition function using forward/backward
                Use forward/backward to compute the derivative (specifically,
                        the derivative of the partition function)
        """
        empirical_features = np.zeros(weights.shape)
        expected_features = np.zeros(weights.shape)
        log_likelihood = 0.0

        if verbose:
            starttime = time.time()
            ab_wrong = 0

        for chain in chains:
            prior_potentials, edge_potentials = self.calc_edge_potentials(chain, weights)
            N, K, W = len(chain), len(prior_potentials), len(weights)

            chain_true_features = np.fromiter(chain.true_features, dtype=np.int32)
            chain_potential = weights[chain_true_features].sum()

            alpha = self.forward(prior_potentials, edge_potentials, N, K)
            beta = self.backward(prior_potentials, edge_potentials, N, K)
            partition_Z = clever_logsumexp(alpha[N-1, :])

            if verbose:
                other_partition_Z = clever_logsumexp(beta[0, :]+prior_potentials)
                Z_diff = partition_Z - other_partition_Z
                if Z_diff > 10**-2:
                    ab_wrong += 1
                    print "Z differences (alpha-beta): %2.7f" % (Z_diff)
                    print "Chain length %s" % N

            log_likelihood += chain_potential - partition_Z

            empirical_features[chain_true_features] += 1.0

            if gradient:
                # E_{f(x,y)} where f is a feature function
                #   = sum_{chains,time} f(t,yp,y)p(yp,y)
                #   and our feature functions are binary (on/off)
                #       are they there or not
                # special case, initial probabilities
                # E_{f(x0,y0)} = p(y0|x0-T)f(t0,y0,x0) for all y0
                marginal = np.exp(beta[0, :] + prior_potentials
                                  - partition_Z).clip(0.0, 1.0)
                init_feats = chain.seq_feats[0]
                for (_, _, y), feats in init_feats.items():
                    expected_features[feats] += marginal[y]



                #alpha_reshaped = alpha.reshape(N, 1, K)
                #beta_reshaped = beta.reshape(N, K, 1)
                #marginals = np.exp(alpha_reshaped + edge_potentials
                #            + beta_reshaped - partition_Z).clip(0.0,1.0)

                for t,t_feats in enumerate(chain.seq_feats[1:], 1):
                    outer_ab = np.add.outer(alpha[t-1, :],beta[t, :])
                    marginal = np.exp(outer_ab + edge_potentials[t, :, :]
                                      - partition_Z).clip(0.0, 1.0)

                    for (_, yp, y), feats in t_feats.items():
                         expected_features[feats] += marginal[yp,y]

                # end inner chain loop
            # end outer chain loop -> next chain

        objective_value = (log_likelihood ) - self.regularizer(weights)
        if verbose: print "End of function (with regularization).  LL(theta)=%s" % objective_value
        if verbose: print "AB wrong %s times" % ab_wrong
        if verbose: print "Iteration running time: %s" % \
                           count_time(time.time()-starttime)
        if gradient:
            #print "\n---\ndebugging gradient:"
            #print empirical_features, expected_features
            gradient_value = (empirical_features - expected_features)
            gradient_value -= self.derived_regularizer(weights)
            #print gradient_value
            #print "debugging statementover\n---\n"
            paired = list(enumerate(zip(gradient_value,empirical_features,expected_features)))
            sorted_pairs = sorted(paired, key=lambda x: x[1][0], reverse=True)
            for i,(g,em,ex) in sorted_pairs[:20]:
                print "%dth item. Represents %s" % ((i+1), self.feature_vocab.lookup(i))
                print "Gradient=%2.5f" %  g
                print "Empirical=%2.5f" %  em
                print "Expected=%2.5f" % ex
            print("---\n--\n-")
            return objective_value, gradient_value

        return objective_value


    def viterbi(self, chain, weights=None):
        if weights is None:
            weights = self.weights
        prior_potentials, edge_potentials = self.calc_edge_potentials(chain, weights)
        N, K = len(chain), len(prior_potentials)
        # yp states will keep the previous best energies
        yp_states = prior_potentials
        # matrix that is long enough fro all of our t and wide enough for all K
        # states
        viterbi_matrix = np.ones((N, K), dtype=np.int32) * -1
        for t in xrange(1, N):
            # for each time step, calculate the best "previous" for every
            # possible state
            # keeping track of the last row of options' scores
            best_yps = np.empty(K)
            for y in xrange(K):
                # at this time point, we are looking for the best state which could have both come from yp and emitted the observed tokens
                # so, we check every possible previous state, and see which one was the best
                # we then update our score to that value, thus ensuring that every row has the best possible score
                # this is the induction step
                # prior energy plus transition energy and emission energy
                running_values = yp_states + edge_potentials[t, :, y]
                viterbi_matrix[t, y] = best = running_values.argmax()
                # update the highest score for y
                best_yps[y] = running_values[best]
            yp_states = best_yps
        last_y = yp_states.argmax()
        best_path = []
        for t in reversed(xrange(N)):
            best_path.append(last_y)
            last_y = viterbi_matrix[t, last_y]
        best_path.reverse()
        return best_path

    def run(self, learner=None, learner_arguments={}):
        if not learner:
            learner = self.structured_perceptron
        learner(*learner_arguments)

    def structured_perceptron(self, rate=0.01, iterations=40, validate_oniter=None, validate_onend=None):
        """ The structured perceptron increments good features and decrements existing features """
        print "Starting Structured Perception with rate: %s, iterations: %s" % (rate, iterations)
        weights = self.weights.copy() # local copy for learning
        old_objective = 0.0
        for i in xrange(iterations):
            print "Starting iteration: %s" % (i + 1)
            weight_start = weights.copy()
            new_objective = self.objective(weights, self.training, False, False)
            print "Objective changed: %2.7f" % (new_objective-old_objective)
            old_objective = new_objective
            for chain in self.training:
                for feature in self.path_features(chain, self.viterbi(chain, weights)):
                    weights[feature] -= rate
                for feature in chain.true_features:
                    weights[feature] += rate
            print "Weights changed: %2.2e" % np.sum(np.abs(weight_start - weights))
            if validate_oniter:
                true_paths = (chain.true_labels for chain in self.data)
                predicted_paths = (
                    self.label_vocab.lookup_many(self.viterbi(chain)) for chain in self.data)
                validate_oniter(true_paths, predicted_paths)
        if validate_onend:
            true_paths = (chain.true_labels for chain in self.data)
            predicted_paths = (
                self.label_vocab.lookup_many(self.viterbi(chain)) for chain in self.data)
            validate_onend(true_paths, predicted_paths)
        self.weights = weights

    def reset_weights(self):
        np.random.seed(103847309)
        W = len(self.feature_vocab)
        self.weights = np.random.uniform(-1,1,size=W)

    def better_reset_weights(self, chains=None):
        if chains is None:
            chains = self.dev
        W = len(self.feature_vocab)
        self.weights = np.zeros(W) + 10**-3
        for chain in chains:
            chain_true_features = np.fromiter(chain.true_features, dtype=np.int32)
            self.weights[chain_true_features] + 10**-3

    def l_bfgs(self, validate_onend=None):
        self.reset_weights()
        def min_func(weights):
            a,b = self.objective(weights, self.training, True, True)
            return -a, -b
        val = fmin_l_bfgs_b(min_func, self.weights, disp=1,approx_grad=False)
        self.weights, _, _ = val

    def gradient_ascent_batch(self, rate=0.0001, iterations=500, validate_oniter=None, validate_onend=None):
        """ The gradient ascent uses the gradient to find parameters. this version does it in batch """

        self.better_reset_weights()
        print "Starting Batch Gradient Ascent with rate: %s, iterations: %s" % (rate, iterations)
        weights = self.weights.copy() # local copy for learning
        old_objective = 0.0
        for i in xrange(iterations):

            print "Starting iteration: %s" % (i + 1)
            new_objective, gradient = self.objective(weights, self.training, True, True)
            weight_start = weights.copy()
            weights += rate * gradient

            print "\nWeights changed: %2.2e" % np.sum(np.abs(weight_start - weights))
            for i,x in sorted(list(enumerate(weights)),key=lambda x:x[1], reverse=True)[:10]:
                print "%sth" % i
                print "Value: %s" % x
                print "Label: %s" % self.feature_vocab.lookup(i)
            print "\n-.-\nLast objective: %2.7f" % old_objective
            print "New objective: %2.7f" % new_objective
            print "Objective changed: %2.7f\n-.-\n" % (new_objective-old_objective)
            old_objective = new_objective

            if validate_oniter:
                true_paths = (chain.true_labels for chain in self.dev)
                predicted_paths = (
                    self.label_vocab.lookup_many(self.viterbi(chain)) for chain in self.dev)
                validate_oniter(true_paths, predicted_paths)

        if validate_onend:
            true_paths = (chain.true_labels for chain in self.dev)
            predicted_paths = (
                self.label_vocab.lookup_many(self.viterbi(chain)) for chain in self.dev)
            validate_onend(true_paths, predicted_paths)

        self.weights = weights

    @staticmethod
    def path_features(chain, path):
        """
         Given a path and a chain
         return the sets of features present at the path through the chain
        """
        features = list(chain[0, None, path[0]])
        features.extend(feature_index for t in range(1, len(chain))
                        for feature_index in chain[t, path[t - 1], path[t]])
        return features

    @staticmethod
    def preprocess(raw_data):
        """
        This preprocessing serves two purposes:
            1. Make the attributes/feature engineer for each of the emission tokens
            2. Make the label and feature domain to just the things that are observed.
        For the vocabs, freeze means it'll throw errors if we go out of label domain
        for the features, it'll return the integer mappings for things we've already stored or None if not present
        """
        print "in preprocess"
        label_vocab = Vocabulary()
        feature_vocab = Vocabulary()
        data = []
        for chain_data in raw_data:
            # Make it a chain object
            cur_chain = Chain(chain_data)
            if len(cur_chain) <= 2:
                continue
            # get all of the features out of this chain into the alphabet
            label_vocab.add_many(cur_chain.true_labels)
            feature_vocab.add_many(
                cur_chain(0, None, cur_chain.true_labels[0]))
            for t, prev_label, label in cur_chain:
                feature_vocab.add_many(cur_chain(t, prev_label, label))
            data.append(cur_chain)
        label_vocab.freeze()
        feature_vocab.stop_growth()
        print 'finish'
        return data, label_vocab, feature_vocab

    @staticmethod
    def make_feature_tensors(chains, K, W):
        """
            Turn the features into a binary matrix for faster learning
        """
        sizes = []
        for chain in chains:
            # for garbage collecting scope of extra variables
            sizes.append(LinearCRF.make_indiv_tensor(chain, K, W))
            print sum(sizes), "is the size so far"
        # print sum(sizes), "is the size!"

    @staticmethod
    def make_indiv_tensor(chain, K, W):
        N = len(chain)
        feature_tensor = np.zeros((N, K, K, W))
        f_table = chain.feature_table
        for index,sparse_vector in chain.feature_table.items():
            verbose_vector = np.zeros(W)
            verbose_vector[sparse_vector] = 1
            feature_tensor[index] = verbose_vector
        return feature_tensor

    # ///
    # Debugging Methods Section
    # The following methods are used for debugging things such as the gradient
    # ///

    def gradient_check(self, theta=None):
        """
            Checking the gradient is an important step in implementing an algorithm
            which uses an objective function for optimization

            Taken from http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/

            Notation note:
                I use weights throughout this file to refer to the parameters
                the tutorial above uses 'theta' so I will use theta in this function

            Procedure:
                From the definition of gradients, we know that
                    dJ(theta) = limit      J(theta + eps) - J(theta - eps)
                    d(theta)   (eps->0)                2 * eps

                where eps is a small number, we use 10**-4. it should be small
                but not too small that it causes underflow errors

                So, our gradient function should be roughly equivalent to the rhs

                For multiple parameter gradients, we need:
                pos_theta(i) = theta + epsilon * basis_vector(i)
                where a basis vector is a vector with all zeros but a 1 in ith position
                and similarly
                neg_theta(i) = theta - epsilon * basis_vector(i)

                so then if our gradient function is g and likelihood equation is J
                g_i(theta) \sim J(pos_theta_i) - J(neg_theta_i)
                                   2 * eps
        """
        if theta is None:
            self.gradient_ascent_batch(rate=0.01, iterations=6)
            theta = self.weights

        K = len(theta)
        epsilon = 10**-4
        basis = np.eye(K)

        basis_vector = lambda i: basis[i,:]
        pos_theta = lambda i, theta: theta.copy() + epsilon * basis_vector(i)
        neg_theta = lambda i, theta: theta.copy() - epsilon * basis_vector(i)

        # J = objective function
        pos_J = lambda i, theta: self.objective(pos_theta(i, theta),
                                                self.dev, False, False)
        neg_J = lambda i, theta: self.objective(neg_theta(i, theta),
                                                self.dev, False, False)

        approx_theta = np.zeros(len(theta))
        _, est_theta = self.objective(theta, self.dev, True, False)

        for i in range(len(theta)):
            approx_theta[i] = (pos_J(i, theta) - neg_J(i, theta)) / (2 * epsilon)
            print "%s of %s thetas approximated" % (i+1, K)
            print "This theta_i corresponds to: %s" % self.feature_vocab.lookup(i)
            print "Estimated is our formula, approximated is finite difference"
            print "(Est.) %2.7f - (Approx) %2.7f" % (est_theta[i], approx_theta[i])
            print "\t = %2.7f" % (est_theta[i]-approx_theta[i])
            print "---\n--\n\n"
        # see note 2.

        print "Max difference: %2.7f" % abs(approx_theta - est_theta).max()
        print "Average difference: %2.7f" % (approx_theta - est_theta).mean()
        print "Min different: %2.7f" % abs(approx_theta - est_theta).min()





class Chain(object):

    """
    Notes on the use:
        Attributes:
            True labels: ground truth of the latent variables (the labels)
            tokens: the observable sequence of items, words in this instance
            feature_table: stores which features are present at which edges
        Functions:
            make_token_attributes: calculate the attributes for a token
            make_features: a generator for all features given an input. Note, a feature is between two nodes, or a prior on the latent varibale

        (chain(x)) Calling a chain object will make features for the arguments

        (chain[k]) Using the chain as a list will set or retrieve from the
                   feature table

        (for x in chain)  Using the chain as an iterator will iterate through
                          the truth labels

        Notation: I use y and yp for the indices of label and prev_label respectively
    """

    def __init__(self, chain_data):
        """
            Input: Raw chain data in the form [(observed variable,true label variable), ...]
            Process:
                Populate attribute table
                create background variables
                set the stage for further processing with empty variables
        """
        self.N = N = len(chain_data)
        self.tokens, self.true_labels = reversezip(chain_data)
        self.attributes = [[] for i in xrange(N)]
        self.feature_table = {}
        self.make_token_attributes()
        self.true_features = None
        self.seq_feats = []

    @classmethod
    def from_vault(cls, storage_items):
        chain_data, feature_table, true_features, seq_feats = storage_items
        newchain = cls(chain_data)
        newchain.feature_table = feature_table
        newchain.true_features = true_features
        newchain.seq_feats = seq_feats

    def vault_items(self):
        return (zip(self.tokens, self.true_labels),
                self.feature_table,
                self.true_features, self.seq_feats)

    def make_token_attributes(self):
        """
        Calculate attributes for every token in the sequence chain

        Rationale:
            Tokens are the sequence of observable states
            This function calculates the attributes for each of these states
            They are not called features because features are functions on the attributes
        """
        self.attributes[0].append("start_token")
        self.attributes[-1].append("end_token")
        for t, token in enumerate(self.tokens):
            # features go here
            word_feat = lambda w: "word=%s" % w
            is_capitalized = lambda w: (["is_capitalized=True"]
                                        if w[0].isupper() else [])
            non_alphanumerics = lambda w: ["contains(%r)" % c for \
                                            c in re.findall('[^a-zA-Z0-9]', w)]
            # simplified = lambda w: 'simplified=' + \
            #    re.sub(
            #        '[0-9]', '0', re.sub('[^a-zA-Z0-9()\.\,]', '', w.lower()))

            # add them to the list of attributes, indexed by time (the t
            # variable)
            self.attributes[t].append(word_feat(token))
            self.attributes[t].extend(is_capitalized(token))
            self.attributes[t].extend(non_alphanumerics(token))
            # self.attributes[t].append(simplified(token)) #Have this taken out
            # for now

    def make_features(self, t, prev_label, label):
        """  Yield features as combinations of attributes for the indexed triplet """
        for attribute in self.attributes[t]:
            yield "[%s > %s]" % (label, attribute)
        if prev_label is None:
            yield "[%s]" % label
        else:
            yield "[%s + %s]" % (prev_label, label)

    def seq_map_features(self):
        self.seq_feats = [{(t,yp,y):value for (t,yp,y),value
                           in self.feature_table.items() if t==t_i}
                          for t_i in xrange(self.N)]

    def enumerate(self):
        """ Enumerate the chain as a sequence of hidden state bigrams """
        return bigram_enum(self.true_labels)

    def setf(self, t, yp, y, features):
        """ sets the features for a single triplet """
        self.feature_table[t, yp, y] = features

    def getf(self, t, yp, y):
        """ gets the features for a single triplet """
        return self.feature_table[t, yp, y]

    def __call__(self, t, yp, y):
        """ Makes the features for a single triplet, this is a shortcut """
        return self.make_features(t, yp, y)

    def __getitem__(self, k):
        """ chains act as a container type """
        try:
            return self.feature_table[k]
        except KeyError:
            raise (KeyError, "The Linear CRF Chain variable"
                             " is not working properly.")

    def __setitem__(self, k, v):
        """ Emulates container types, sets a feature in the
            feature table for a single triplet """
        self.feature_table[k] = v

    def __iter__(self):
        """ Enumerates the chain for the sequence of
            hidden (true) state bigrams """
        return bigram_enum(self.true_labels)

    def __len__(self):
        return len(self.true_labels)



"""
IMPLEMENTATION NOTES
(I log everything)
===============================================================================
-
--
Note 1. LL Derivation, specifically, the jacobian of the partition function

Since the derivative of the partition function is the expected feature value,
it's a sum over the probability of points and the value at that point
(as expected values typically do)

We use alpha and beta in this calculation. but they are in log space.
using Eq. 26, page 6 in Rahul Gupta's Conditional Random Fields manuscript
we find that sum_t alpha[t-1,:]*Q*beta[t,:] is a chain's contribution
to the expected count.
--
-
===============================================================================
-
--
Note 2. Parallelizing with joblib.
can't pickle functions. but maybe I can make better some day.

def parapprox(i):
    val = (pos_J(i, theta) - neg_J(i, theta)) / (2 * epsilon)
    print "%s of %s thetas approximated" % (i+1, K)
    return val

num_cores = multiprocessing.cpu_count()

approx_theta = Parallel(n_jobs=num_cores)(delayed(parapprox)(i) for i in range(K))
approx_theta = np.array(approx_theta)
--
-
===============================================================================
-
--
Note 3. Forward function

According to Sutton&McCallum, a transition potential is the
# exponentiated sum of features * weights
# since, we have the sum of features * weights, we exponentiate
# since it's a dynamic programming operation, we store the last value
# and reuse it in a bunch of comptuations that depend on it
# we also make use of the logsumexp trick for underflow errors
# and note that exp(log(x)) = x and
# edge_potential for a single t,yp,y is a p(yt|ypt,xt) and
# we deal with log probabilities in the crf (log linear model)
# so we want to exponentiate the edge potentials
# and we use the exp(log(x)) to combine the alpha with the potentials
--
-
===============================================================================
============
SCRAPS ON SCRAPS ON SCRAPS
============


                # See implementation note 1. at the bottom of this document.
                # Using Tim's version, since I'm assuming it's been thoroughly debugged =)
                ""
                for yp in xrange(k):
                    for y in range(k):
                        feature_sums = np.zeros((N,K))
                        feature_sums[chain[:,yp,y]] += 1
                        feature_sums.sum(axis=1)
                ""
                # special case, initial probabilities
                marginal = np.exp(beta[0, :] + prior_potentials
                                  - partition_Z).clip(0.0, 1.0)
                for y in xrange(K):
                    p = marginal[y]
                    if p < 1e-40:
                        continue
                    expected_features[chain[0, None, y]] += p

                # iterate over the chain for each Y, YP value.
                # I might try a faster way. See Note 1 again.
                for t in xrange(1, N):
                    outer_ab = np.add.outer(alpha[t-1, :],beta[t, :])
                    marginal = np.exp(outer_ab + edge_potentials[t, :, :]
                                      - partition_Z).clip(0.0, 1.0)
                    test = np.zeros(())
                    test
                    for yp in xrange(K):
                        for y in xrange(K):
                            p = marginal[yp, y]
                            expected_features[chain[t, yp, y]] += p


                alpha_reshaped = alpha.reshape(N, K, 1)
                beta_reshaped = beta.reshape(N, 1, K)
                log_probs = (alpha_reshaped + edge_potentials
                            + beta_reshaped - partition_Z)
                log_probs = log_probs.reshape(log_probs.shape+(1,))
                feature_tensor = LinearCRF.make_indiv_tensor(chain, K, W)
                expected_features += np.sum(np.exp(log_probs) * feature_tensor,
                                     axis=(0, 1, 2))




                # iterate over the chain for each Y, YP value.
                # I might try a faster way. See Note 1 again.

                alpha_reshaped = alpha.reshape(N, 1, K)
                beta_reshaped = beta.reshape(N, K, 1)
                marginals = np.exp(alpha_reshaped + edge_potentials
                            + beta_reshaped - partition_Z).clip(0.0,1.0)
                # print marginals.max()
                #
                for (t, yp, y), feats in chain.feature_table.items():
                     if t==0: continue
                     expected_features[feats] += marginals[t,yp,y]
                #
                    # expected_features[feats] += 1

                    #MANUAL WAY
                    #for y in xrange(K):
                    #    for yp in xrange(K):
                    #        c=marginal[yp,y]
                    #        for feat in chain[t,yp,y]:
                    #            expected_features[feat] = c
                            # expected_features[chain[t,yp,y]] += marginal[yp,y]
                    # print marginal.max()

"""
