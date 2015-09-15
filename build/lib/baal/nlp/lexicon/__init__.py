import pickle
import os
from generic_rules import generate_rules

curdir = os.path.dirname(os.path.abspath(__file__))

def get_entries(lexical_item):
    return []

def get_wordlists():
    with open(curdir+'/wordlists.pkl', 'rb') as fp:
        return pickle.load(fp)
