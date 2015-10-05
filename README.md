# BAAL (Brian's Arsenal of Algorithms for L*)

A collection of algorithms and data structures I use for research.

So, the arsenal is composed of: 

### Core Algorithms 
* Conditional Random Fields
* Bayesian Hierarchical Clustering/Grouping 
        + A port from a MATLAB port from the original C implementation
* Natural Language Processing
    - Corpora
        + Interfaces and handlers for corpora go here for a variety of datasets
    - Grammars
        + Tree Insertion Grammar
    - Induce
        + Tree enrichment rules from Michael Collins' dissertation. Mostly ported from Stanford CoreNLP
    - Lexicon
        + Words and their properties. used to interface with the grammars eventually
    - Parse
        + Tree chart for the tree insertion grammar
    - Semantics
        + Currently, a simple model inspired by Hobbsian Logical Form
    - Generate
        + Natural Language Generation stuff
    - Note: the tree data structure which bears the brunt of the tree grammar work is in the utils.data_structures

### Useful stuff
* Utils
    - A variety of utilities that I find useful
    - Config
        + Store global configuration modules
    - Data Structures
        + flyweights, singletons, and trees
    - sugar
        + decorators and other syntactic sugar
    - vault
        + model storage
    - montecarlo, vocabulary (borrowed, citation in file), hobbsian logical form, and some other things. 
