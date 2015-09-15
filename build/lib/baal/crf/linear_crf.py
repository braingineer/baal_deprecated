"""
Linear Chain Conditional Random Field
Adapted from Tim Vieira (https://github.com/timvieira/crf)
"""

from baal.utils.general import bigram_enum, reversezip
from baal.utils.data import split_list
from baal.utils.vocabulary import Vocabulary
from scipy.misc import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from collections import defaultdict
import numpy as np
import re

def clever_logsumexp(some_array):
    the_trick = some_array.max()
    return the_trick+logsumexp(some_array-the_trick)

class LinearCRF(object):

    def __init__(self, data, label_vocab, feature_vocab, split_proportions=(0.7, 0.15, 0.15)):

        self.training, self.dev, self.test = split_list(
            data, split_proportions)
        self.label_vocab = label_vocab
        self.feature_vocab = feature_vocab
        self.weights = np.zeros(len(feature_vocab))

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
            for t in xrange(len(chain)):
                for y, label in label_vocab.enum():
                    for yp, prev_label in label_vocab.enum():
                        # store the list of feature indices in the look up
                        # table
                        chain[t, yp, y] = np.fromiter(
                            fmap(chain(t, prev_label, label)), dtype=np.int32)
            chain.true_features = LinearCRF.path_features(
                chain, label_vocab.map(chain.true_labels))
        return LinearCRF(data, label_vocab, feature_vocab)

    def calc_edge_potentials(self, chain, weights):
        """
        assuming self.weights has a certain value
        """
        # make the prior edge potentials (0,None,y)
        N, K = len(chain), len(self.label_vocab)
        prior_potentials = np.array(
            [weights[chain[0, None, y]].sum() for y in xrange(K)])
        edge_potentials = np.empty((N, K, K))
        # t starts at 1 because prior potentiasl are our 0 state.
        for t in xrange(1, N):
            for y in xrange(K):
                for yp in xrange(K):
                    edge_potentials[t, yp, y] = weights[chain[t, yp, y]].sum()
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
                # According to Sutton&McCallum, a transition potential is the
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
                alpha[t,y] = clever_logsumexp(prior_alpha + edge_potentials[t, :, y])
        return alpha

    def backward(self, edge_potentials, N, K):
        """
            Calculates the probability of the data.
            Input:
                edge_potentials = energy for all t,yp,y triplets
                N = size of chain
                K = size of latent state space

            Note: this is a log beta. We keep it in log space for tractibilty.
        """
        beta = np.zeros((N,K))
        for t in reversed(xrange(N)):
            last_beta = beta[t+1, :]
            for yp in xrange(K):
                beta[t,yp] = clever_logsumexp(last_beta + edge_potentials[t,yp,:])
        return beta

    def loglikelihood_and_derivative(self, weights, chains):
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


        for chain in chains:
            prior_potentials, edge_potentials = self.calc_edge_potentials(chain, weights)
            N, K = len(chain), len(prior_potentials)

            chain_true_features = np.fromiter(chain.true_features, dtype=np.int32)
            chain_potential = weights[chain_true_features].sum()

            alpha = self.forward(prior_potentials, edge_potentials, N, K)
            beta = self.backward(prior_potentials, edge_potentials, N, K)
            partition_Z = clever_logsumexp(alpha[N-1, :])

            log_likelihood += chain_potential - partition_Z
            empericial_features[chain_true_features] += 1

            # ///
            # Calculating the derivative of the partition function.
            # There might be a faster way
            # Oh well
            # Basically:
            # Since the derivative of the partition function is the expected
            # feature value, it's a sum over the probability of points and the value
            # at that point (as expected values typically do)
            # We use alpha and beta in this calculation.
            # but they are in log space.
            # using Eq. 26, page 6 in Rahul Gupta's Conditional Random Fields manuscript
            # we find that sum_t alpha[t-1,:]*Q*beta[t,:] is a chain's contribution
            # to the expected count
            # Using Tim's version, since I'm assuming it's been thoroughly debugged =)

            #special case, initial probabilities
            marginal = np.exp(prior_potentials + beta[0, :] - partition_Z).clip(0.0, 1.0)

            for t in xrange(1, N):
                marginal = np.exp((np.add.outer( alpha[t-1, :],
                                                    beta[t, :]),
                                  + g[t, :, :] - partition_Z)).clip(0.0, 1.0)

                for yp in xrange(K):
                    for y in xrange(K):
                        p = c[yp, y]
                        if p < 1e-40:
                            continue  # skip small updates
                        expected_features[chain[t, yp, y]] += p

                # end inner chain loop
            # end outer chain loop -> next chain
        returned_ll = -log_likelihood - self.regularizer(weights)
        returned_deriv_ll =   -(emperical_features - expected_features -
                                 self.derived_regularizer(weights))

        return returned_ll, returned_deriv_ll

    def viterbi(self, chain, weights):
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
        for i in xrange(iterations):
            print "Starting iteration: %s" % (i + 1)
            weight_start = weights.copy()
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

    def l_bfgs(self, validate_onend=None):
        min_func = lambda weights: loglikelihood_and_derivative(weights, self.training)
        val = fmin_l_bfgs_b(min_func, np.random.randn(self.weights.shape))
        print val
        self.weights, _, _ = val

    @staticmethod
    def path_features(chain, path):
        # Given a path and a chain
        # return the sets of features present at the path through the chain
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
        label_vocab = Vocabulary()
        feature_vocab = Vocabulary()
        data = []
        for chain_data in raw_data:
            # Make it a chain object
            cur_chain = Chain(chain_data)
            # get all of the features out of this chain into the alphabet
            label_vocab.add_many(cur_chain.true_labels)
            feature_vocab.add_many(
                cur_chain(0, None, cur_chain.true_labels[0]))
            for t, prev_label, label in cur_chain:
                feature_vocab.add_many(cur_chain(t, prev_label, label))
            data.append(cur_chain)
        label_vocab.freeze()
        feature_vocab.stop_growth()
        return data, label_vocab, feature_vocab


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

        Calling a chain object will make features for the arguments
        Using the chain as a list will set or retrieve from the feature table
        Using the chain as an iterator will iterate through the truth labels

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
            is_capitalized = lambda w: "is_capitalized=True" if w[
                0].isupper() else "is_capitalized=False"
            non_alphanumerics = lambda w: [
                "contains(%r)" % c for c in re.findall('[^a-zA-Z0-9]', w)]
            # simplified = lambda w: 'simplified=' + \
            #    re.sub(
            #        '[0-9]', '0', re.sub('[^a-zA-Z0-9()\.\,]', '', w.lower()))

            # add them to the list of attributes, indexed by time (the t
            # variable)
            self.attributes[t].append(word_feat(token))
            self.attributes[t].append(is_capitalized(token))
            self.attributes[t].extend(non_alphanumerics(token))
            # self.attributes[t].append(simplified(token)) #Have this taken out
            # for now

    def make_features(self, t, prev_label, label):
        """  Yield features as combinations of attributes for the indexed triplet """
        for attribute in self.attributes[t]:
            yield "[%s - %s]" % (label, attribute)
        yield "[%s]" % label
        yield "[%s - %s]" % (prev_label, label)

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
