import pickle
import os
from generic_rules import generate_rules

curdir = os.path.dirname(os.path.abspath(__file__))

def get_entries(lexical_item):
    return []

def get_wordlists():
    with open(curdir+'/wordlists.pkl', 'rb') as fp:
        return pickle.load(fp)

def add_word(word, pos):
    wordlist = get_wordlists()
    if pos not in wordlist.keys():
        print "oops, pos not in. pick one of %s" % wordlist.keys()
    if word not in wordlist[pos]:
        wordlist[pos].append(word)
    else:
        print "not added. already in"
    with open(curdir+'/wordlists.pkl', 'wb') as fp:
        pickle.dump(wordlist, fp)
