# Brian's Arsenal of Algorithms for [Language|Learning|Life] 

A collection of algorithms and data structures I use for research.

So, the arsenal is composed of: 

-------
* CRF
    - Linear CRF
    - (there's an hmm here)

-------
* learn 
    - Bayesian Hierarchical Grouping 
        + A port from a MATLAB port from the original C implementation
* NLP
    - Corpora
        + Interfaces and handlers for corpora go here
    - Grammars
        + As of right now, just a couple toy / made up grammars
    - Induce
        + Tree enrichment rules from Michael Collins' dissertation. Mostly ported from Stanford CoreNLP
    - Lexicon
        + Words and their properties. used to interface with the grammars eventually
    - Parse
        + Tree chart
    - Semantics
        + Currently, a simple model inspired by Hobbsian Logical Form
    - Note: the tree data structure which bears the brunt of the parsing work is in the utils.data_structures

-------
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

-------

* Future plans
    - Various grammars (CFG,TSG,and TAG)
    - Logical formalisms (Hobbsian probably)
    - Wrapper around scientific resources, like the [Scikit laboratory](http://scikit-learn-laboratory.readthedocs.org/en/latest/)
