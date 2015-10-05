# baal

This package contains the collection of algorithms and data structure I use for research.   Hence, the name is an acronym: *brian's arsenal algorithms for language/learning*. 

## Contents 
* **.crf**: Conditional Random Field
        + currently, only a linear chain implementation
* **.learn.bhc**: Bayesian Hierarchical Clustering/Grouping 
        + A port from a MATLAB port from the original C implementation
* **.nlp**: Natural Language Processing
    - **.nlp.corpora**:
        + Interfaces and handlers for corpora go here for a variety of datasets
    - **.nlp.grammars**: 
        + Tree Insertion Grammar
    - **.nlp.induce**:
        + Tree enrichment rules from Michael Collins' dissertation. Mostly ported from Stanford CoreNLP
    - **.nlp.lexicon**:
        + Words and their properties. 
    - **.nlp.parse**:
        + CKY Tree chart parser for the tree insertion grammar
    - **.nlp.semantics**:
        + a simple model of hobbsian logical form that uses derived tree structure to produce logical predicates
    - **.nlp.generate**:
        + a simple model of natural language generation
    - Note: the tree data structure which bears the brunt of the tree grammar work is in the utils.data_structures
* **.utils**
    - **.general**:
        + useful utilities to make algorithms nicer
    - **.utils.config**:
        + Store global configuration modules
    - **.utils.config**
        + flyweights, singletons, and trees
    - **.utils.sugar**
        + decorators and other syntactic sugar
    - **.utils.vault**
        + model storage
    - **.montecarlo**
    - **.vocabulary** (borrowed, citation in file)
* **.science**
    - experiment scripts for reproducability 
