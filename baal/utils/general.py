from time import time
import collections
from math import floor
import types


class SimpleProgress:
    """
    A progress function that does linear extrapolation between
      time spent and number of things completed.

    Usage:
      prog = SimpleProgress(number_of_units)
      prog.start()
      for i in range(num_iterations):
        prog.update(i)
        is this working
    """

    def __init__(self, total):
        self.total = total
        self.count = 0
        self.num_updates = 0

    def start_progress(self):
        self.start_time = time()

    def incr(self, amount=1):
        self.count+=1

    def should_output(self, trigger_percent=0.1):
        if self.count > self.total * trigger_percent * self.num_updates:
            self.num_updates += 1
            return True
        return False

    def update(self, x = None):
        if x is None:
            x = self.count

        if x > 0:
            elapsed = time() - self.start_time
            percDone = x * 100.0 / self.total
            estimatedTimeInSec = (elapsed / float(x)) * self.total
            return """
                  %s %s percent
                  %s Processed
                  Elapsed time: %s
                  Estimated time: %s
                  --------""" % (self.bar(percDone),
                                 round(percDone, 2),
                                 x, self.form(elapsed),
                                 self.form(estimatedTimeInSec))
        return ""

    def expiring(self):
        elapsed = time() - self.start_time
        return elapsed / (60.0 ** 2) > 71.

    def form(self, t):
        hour = int(t / (60.0 * 60.0))
        minute = int(t / 60.0 - hour * 60)
        sec = int(t - minute * 60 - hour * 3600)
        return "%s Hours, %s Minutes, %s Seconds" % (hour, minute, sec)

    def bar(self, perc):
        done = int(round(30 * (perc / 100.0)))
        left = 30 - done
        return "[%s%s]" % ('|' * done, ':' * left)


def process_file(filename):
    fp = open(filename)
    data = [x.replace("\n", "") for x in fp.readlines()]
    fp.close()
    return data


def bigram_enum(sequence):
    for i, x in enumerate(sequence[1:], 1):
        yield i, sequence[i - 1], x

def backward_enumeration(thelist):
   for index in reversed(xrange(len(thelist))):
      yield index, thelist[index]

def reversezip(xys):
    return [[x[i] for x in xys] for i in range(len(xys[0]))]


def lrange(x, start=0, step=1):
    return range(start, len(x), step)


def get_ind(x, i):
    return [y[i] for y in x]


def reverse_dict(in_dict):
    out_dict = {n: k for k, ns in in_dict.items() for n in ns}
    return out_dict


def empty(x):
    if isinstance(x, collections.Iterable):
        return not len(x) > 0
    raise TypeError


def flatten(some_list):
    """
    assumes list of lists
    for sublist in some_list:
        for item in sublist:
            yield item

    """
    return [item for sublist in some_list for item in sublist]


def count_time(t, d=2):
    seconds = t % 60
    minutes = int(floor(t / 60))
    if d == 2:
        return "{0:02} Min; {1:2.2f} Sec.".format(minutes, seconds)
    raise NotImplementedError


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    index = {'0':HEADER,
             '1':OKBLUE,
             '2':OKGREEN,
             "w":WARNING,
             "f":FAIL,
             "b":BOLD,
             "u":UNDERLINE}


def cprint(x,levels=[0]):
    """ Deprecated for colorama """
    print "".join([bcolors.index[str(level)]
                   for level in levels]) + x + bcolors.ENDC


def cformat(x, levels=[0]):
    return "".join([bcolors.index[str(level)]
                   for level in levels]) + x + bcolors.ENDC


def cprint_showcase():
    print bcolors.BOLD + bcolors.UNDERLINE + "Options Showcase" + bcolors.ENDC
    for name in bcolors.index.keys():
        cprint("Argument option: %s.  Effect shown." % name, [name])

def nonstr_join(arr,sep):
    if not isinstance(arr,types.ListType):
        arr = [arr]
    return str(sep).join([str(x) for x in arr])

class while_loop_manager(object):
    def __init__(self, condition, log_on=False):
        self.condition = condition
        self.iter_i = 0
        self.log_on = log_on

    def loop(self,loop_indicator):
        while self.condition(loop_indicator):
            if self.log_on:
                self.log()
            self.iter_i += 1
            yield True
        yield False

    def log(self):
        print "Iteration %s: %d" % self.iter_i
