from baal.utils.general import SimpleProgress
import random
from time import time
from numpy.random import sample


class Markov_Chain_Monte_Carlo:
    # ///////////////////
    # Component-Wise MCMC
    # ///////////////////
    """
    Procedure:
    set t=1
    generate initial values, set Theta(t)=inital values
    while t<T
       t=t+1
       for each theta_i in Theta(t-1):
           generate a proposal theta_i*
           evaluate the acceptance probability; a=min(1,evaluation())
           generate a u from the uniform distribution
           if u <= a, accept
    """
    def __init__(self, data):
        self.data = data
        self.performance_log = [[], []]

    def evaluation(self, param_proposal, param_current, data):
        # This should be implemented by children
        pass

    def initial(self, data):
        # This should be implemented by children
        pass

    def proposal(self, parameters, current, i):
        # This should be implemented by children
        pass

    def run(self, T, name=None, prog_frequency=100):
        # Initialize Parameters
        parameters = [self.initial()]
        progress = SimpleProgress(T)
        progress.start_progress()
        print "Initialized, Locked and Loading, Prepare to rock"
        num_rejected=[0 for _ in range(len(parameters[0]))];
        for t in range(T):
            self.alpha = self.estimate_availability(parameters[-1])
            if t%prog_frequency==0:
                if name:
                    print "------\n%s's MCMC below\n------" % name
                print progress.update(t)
            #push last iteration's parameters forward; this put it into t+1
            parameters.append([x for x in parameters[-1]])
            #wiggle each param

            random.seed(time.time())
            i_p = range(len(parameters[t]))
            random.shuffle(i_p)
            for i in i_p:
                #wiggle wiggle via gaussian random walk
                param_proposal = self.proposal(parameters[t+1], parameters[t][i], i)
                #print "proposed %s over %s, param #%s" % (param_proposal, parameters[t][i],i)

                ### NOTICE:  this is a log space evaluation.
                ### TODO: parameterize to choose between log space evaluation and normal evaluation
                alpha = min( [1, (self.evaluation(parameters[t+1][:i] + [param_proposal]+parameters[t][i+1:],
                                                  parameters[t+1], i))])
                u = sample(1)[0]

                #accept or reject
                if u<=alpha: parameters[t+1][i]=param_proposal
                else: num_rejected[i]+=1
                self.performance_log[0].append([alpha, 1 if u<=alpha else 0])
            if progress.expiring():
                return parameters, self.alpha, num_rejected


        return parameters, self.alpha, num_rejected
