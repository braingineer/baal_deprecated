import random
import numpy as np
from math import log,exp
from numpy.random import random_sample 
import matplotlib.pyplot as plt
import pprint
from baal.utils import get_ind

class hmm(object): 
    def __init__(self, emissions, transitions, initial=None):
        self.emissions = emissions
        self.transitions = transitions
        if not initial:
            initial = {k:1./len(transitions.keys()) for k in transitions.keys()}
        self.initial = initial
        
        
    def viterbi(self, observations):
        #Given:
        # Observations
        # a model consisting of a transition, emission table
        # the last element in the transitions table is the initial probabilities
        # otherwise, we look up transitions as transitions[from][to] 
        # for instance, it could be transitions['sunny']['sunny']
        # we look up emissions as emissions[state][observed variable value]
        # for instance, it could be emissions['sunny']['umbrella']
        #the table we use for tracking probabilities
        #row is latent variable, column is the possible values it can take
        states = self.transitions.keys()
        safe_log = lambda x: log(x) if x>0 else float('-inf')
        arg_max = lambda states: sorted(enumerate(states),key=lambda state:state[1][0],reverse=True)[0]
        transitions,emissions = self.transitions,self.emissions
        num_states = len(transitions)
        var_state_table = [[(float('-inf'),None) for i in xrange(num_states)] for j in xrange(len(observations))]
        
        #initialize the first latent variable. its prior probability usually a uniform
        var_state_table[0] = [(safe_log(self.initial[state])+log(emissions[state][observations[0]]),None,state) for state in states]
        print log(emissions['sunny']['no-umbrella'])
        print emissions
        for x_i,observed in enumerate(observations[1:],1): #for each latent variable after the initial and each observation
            for cur_k,cur_state in enumerate(states): #for each state in the current latent variable
                best_prob,best_kindex = float('-inf'),None #trying to find the best predecessor, so make this the worst possible
                for prev_k,prev_state in enumerate(states): #for each state in the latent variable right before this one
                    prev_k_prob,prev_kindex,prev_beststate = var_state_table[x_i-1][prev_k] #look up the previous latent variable's state
                    transition_prob = log(transitions[prev_state][cur_state]) #the transition from the previous state to this state
                    if prev_k_prob + transition_prob > best_prob: #we have found a better path for us
                        best_prob = prev_k_prob + transition_prob
                        best_kindex = prev_k
                
                var_state_table[x_i][cur_k] = (log(emissions[cur_state][observed])+best_prob,best_kindex,states[best_kindex])
        import pprint
        pp=pprint.PrettyPrinter(indent=3)
        pp.pprint(var_state_table)
        #begin the backwards pointer accumulation
        path = [None for x in xrange(len(observations))]
        path[-1] = arg_max(var_state_table[-1])
    
        for i in range((len(path)-1), 0,-1):
            print path[i]
            state_i, (prob,prev_k,prevstate) = path[i]
            path[i-1]=(prev_k,(var_state_table[i-1][prev_k]))
    
        return path

    @staticmethod
    def sample(dict_dist):
        init_arr = dict_dist.items()
        bins = np.add.accumulate(np.array([x[1] for x in init_arr]))
        inds = np.digitize(random_sample(1),bins)
        return init_arr[inds][0]

    def sample_observations(self,max_o=100,with_truth=True):
        state = self.sample(self.initial)
        for o_i in xrange(max_o):
            last_state = state; state = self.sample(self.transitions[state]);
            yield self.sample(self.emissions[last_state]) if not with_truth else (last_state,self.sample(self.emissions[last_state]))
        return
    
    @staticmethod
    def mle_training(n_observations):
        #going to assume observations are supervised
        # i.e. they are (y,x) where y is the latent state for observed x
        
        #Build the data structures and methods
        #Notice that we smooth by adding 1.0 counts to everything. 
        latent_states = set(get_ind(n_observations[0],0))
        observed_states = set(get_ind(n_observations[0],1))
        emissions = {latent:{observed:1.0 for observed in observed_states} for latent in latent_states}
        transitions = {latent:{latent:1.0 for latent in latent_states} for latent in latent_states}
        initial = {latent:0.0 for latent in latent_states}
        
        count_transitions = lambda obs,x,y=None: sum([1.0 for i,o in enumerate(obs[:-1]) if (o[0]==x and obs[i+1][0]==y if y else o[0]==x)])
        count_emissions = lambda obs,x,y=None: sum([1.0 for i,o in enumerate(obs) if (o[0]==x and o[1]==y if y else o[0]==x)])
        normalize_counts = lambda some_dict: {key:(count/sum(some_dict.values())) for key,count in some_dict.items()}
        
        #Go through the data and count occurrences 
        for obs in n_observations:
            initial[obs[0][0]]+=1.0
            for latent in latent_states:
                for observed in observed_states:
                    emissions[latent][observed]+=count_emissions(obs,latent,observed)
                for latent_next in latent_states:
                    transitions[latent][latent_next] = count_transitions(obs,latent,latent_next)
        
        #Normalize to probabilities
        emissions = {latent:normalize_counts(emits) for latent,emits in emissions.items()}
        transitions = {latent:normalize_counts(nexts) for latent,nexts in transitions.items()}
        initial = normalize_counts(initial)
        return hmm(emissions,transitions,initial)
        
        
    @staticmethod
    def example_hmm():
        emissions = {"sunny":{"umbrella":0.1,"no-umbrella":0.9}, 
                     "cloudy":{"umbrella":0.4,"no-umbrella":0.6},
                     "rainy":{"umbrella":0.9,"no-umbrella":0.1}}
        transitions= {"sunny": {"sunny": 0.75,"cloudy": 0.15,"rainy": 0.1},
                      "cloudy": {"sunny": 0.1,"cloudy": 0.7,"rainy": 0.2},
                      "rainy": {"sunny": 0.1,"cloudy": 0.1,"rainy": 0.8}}
        states = transitions.keys()
        initial = {k:1./len(transitions) for i,k in enumerate(transitions.keys())}
        return hmm(emissions,transitions,initial)
    
    
def experiment_one(hmm_ex=None,num_obvs=20,print_result=True):
    states = ["sunny", "cloudy", "rainy"]
    if not hmm_ex: hmm_ex = hmm.example_hmm()
    better_observations =  [x for x in hmm_ex.sample_observations(num_obvs)]
    correct = 0
    path =  hmm_ex.viterbi(get_ind(better_observations,1))
    if print_result:
        for i,(p,o) in enumerate(zip(path,better_observations)):
            s_i,(prob,prev_k,prevstate) = p
            prob = exp(prob)
            state = states[s_i]
            if state==better_observations[i][0]: correct+=1.0
            fixed_width_o = ( "%s) =" % (o[1]) ).rjust(14)
            print "P(%s | %s  %2.1e" % ((state).ljust(7),fixed_width_o,prob)
    print "%2.2f States Correctly Inferred" % (correct/num_obvs)
    
def experiment_two(a_hmm=None,num_obvs=50,num_trials=100):
    states = ["sunny", "cloudy", "rainy"]
    if not a_hmm: a_hmm = hmm.example_hmm()
    hmm_performance = []
    baseline_performance = []
    smarter_baseline_performance = []
    smarter_baseline_map = {'umbrella': 'rainy', 'no-umbrella': 'sunny'}
    for t in xrange(num_trials):
        observations = [x for x in a_hmm.sample_observations(num_obvs)]
        path =  a_hmm.viterbi(get_ind(observations,1))
        hmm_correct = sum([1.0 for i,p in enumerate(path) if states[p[0]]==observations[i][0]])/num_obvs
        baseline_correct = sum([1.0 for i,p in enumerate(path) if hmm.sample(a_hmm.initial)==observations[i][0]])/num_obvs
        smarter_baseline_correct = sum([1.0 for i,p in enumerate(path) if smarter_baseline_map[observations[i][1]]==observations[i][0]])/num_obvs
        hmm_performance.append(hmm_correct)
        baseline_performance.append(baseline_correct)
        smarter_baseline_performance.append(smarter_baseline_correct)
    
    #hmm_performance,baseline_performance,smarter_baseline_performance = experiment_two(num_obvs=100,num_trials=1000)
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.hist(hmm_performance,bins=20,label='Viterbi',alpha=0.7)
    ax1.hist(baseline_performance,bins=20,label='Random Chance (baseline)',alpha=0.7)
    ax1.hist(smarter_baseline_performance,bins=20,label='Deterministic (baseline)',alpha=0.7)
    plt.legend()
    plt.show()
    #return hmm_performance, baseline_performance, smarter_baseline_performance