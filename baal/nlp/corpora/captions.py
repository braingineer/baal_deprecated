from __future__ import print_function
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
import logging
from baal.utils.data_structures import trees

base_fp = "/home/cogniton/research/data"

coco_trees = "{}/coco_parses".format(base_fp)
flickr8k_trees = "{}/flickr8k".format(base_fp)
flickr30k_trees = "{}/flickr30k".format(base_fp)

sentence_file = "sentdump.txt"
stanford_parsefile = "parsed_captions.txt"
tree_file = "saved_elemtrees.pkl"

def gettrees(datasetname):
    """ get the unpickled trees object.
      the format for the trees objects is:
                  {key:value,...} where key is a bracketed tree
                                        and value is a count"""
    with open({"coco": lambda: "{}/{}".format(coco_trees, tree_file),
               "8k": lambda: "{}/{}".format(flickr8k_trees, tree_file),
               "30k": lambda: "{}/{}".format(flickr30k_trees, tree_file)}[datasetname]()) as fp:
        return pickle.load(fp)

def getsents(datasetname):
    with open({"coco": lambda: "{}/{}".format(coco_trees, sentence_file),
               "8k": lambda: "{}/{}".format(flickr8k_trees, sentence_file),
               "30k": lambda: "{}/{}".format(flickr30k_trees, sentence_file)}[datasetname]()) as fp:
        return [x.replace("\n","") for x in fp.readlines()]


def get_all_sentences():
    sents8k = getsents("8k")
    sents30k = getsents("30k")
    sentscoco = getsents("coco")
    print("{} sentences for flickr8k,".format(len(sents8k)),
          "{} sentences for flickr30k,".format(len(sents30k)),
          "and {} sentences for coco".format(len(sentscoco)))
    return sents8k, sents30k, sentscoco

def get_all_treestrings():
    trees8k = gettrees("8k")
    trees30k =  gettrees("30k")
    treescoco = gettrees("coco")
    print("{} trees for flickr8k,".format(len(trees8k)),
          "{} trees for flickr30k,".format(len(trees30k)),
          "and {} trees for coco".format(len(treescoco)))

    return trees8k, trees30k, treescoco

