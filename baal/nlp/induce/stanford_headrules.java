
nonTerminalInfo = Generics.newHashMap();
// This version from Collins' diss (1999: 236-238)
nonTerminalInfo.put("ADJP", new String[][]{{"left", "NNS", "QP", "NN", "$", "ADVP", "JJ", "VBN", "VBG", "ADJP", "JJR", "NP", "JJS", "DT", "FW", "RBR", "RBS", "SBAR", "RB"}});
nonTerminalInfo.put("ADVP", new String[][]{{"right", "RB", "RBR", "RBS", "FW", "ADVP", "TO", "CD", "JJR", "JJ", "IN", "NP", "JJS", "NN"}});
nonTerminalInfo.put("CONJP", new String[][]{{"right", "CC", "RB", "IN"}});
nonTerminalInfo.put("FRAG", new String[][]{{"right"}}); // crap
nonTerminalInfo.put("INTJ", new String[][]{{"left"}});
nonTerminalInfo.put("LST", new String[][]{{"right", "LS", ":"}});
nonTerminalInfo.put("NAC", new String[][]{{"left", "NN", "NNS", "NNP", "NNPS", "NP", "NAC", "EX", "$", "CD", "QP", "PRP", "VBG", "JJ", "JJS", "JJR", "ADJP", "FW"}});
nonTerminalInfo.put("NX", new String[][]{{"left"}}); // crap
nonTerminalInfo.put("PP", new String[][]{{"right", "IN", "TO", "VBG", "VBN", "RP", "FW"}});
// should prefer JJ? (PP (JJ such) (IN as) (NP (NN crocidolite)))
nonTerminalInfo.put("PRN", new String[][]{{"left"}});
nonTerminalInfo.put("PRT", new String[][]{{"right", "RP"}});
nonTerminalInfo.put("QP", new String[][]{{"left", "$", "IN", "NNS", "NN", "JJ", "RB", "DT", "CD", "NCD", "QP", "JJR", "JJS"}});
nonTerminalInfo.put("RRC", new String[][]{{"right", "VP", "NP", "ADVP", "ADJP", "PP"}});
nonTerminalInfo.put("S", new String[][]{{"left", "TO", "IN", "VP", "S", "SBAR", "ADJP", "UCP", "NP"}});
nonTerminalInfo.put("SBAR", new String[][]{{"left", "WHNP", "WHPP", "WHADVP", "WHADJP", "IN", "DT", "S", "SQ", "SINV", "SBAR", "FRAG"}});
nonTerminalInfo.put("SBARQ", new String[][]{{"left", "SQ", "S", "SINV", "SBARQ", "FRAG"}});
nonTerminalInfo.put("SINV", new String[][]{{"left", "VBZ", "VBD", "VBP", "VB", "MD", "VP", "S", "SINV", "ADJP", "NP"}});
nonTerminalInfo.put("SQ", new String[][]{{"left", "VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ"}});
nonTerminalInfo.put("UCP", new String[][]{{"right"}});
nonTerminalInfo.put("VP", new String[][]{{"left", "TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG", "VBP", "AUX", "AUXG", "VP", "ADJP", "NN", "NNS", "NP"}});
nonTerminalInfo.put("WHADJP", new String[][]{{"left", "CC", "WRB", "JJ", "ADJP"}});
nonTerminalInfo.put("WHADVP", new String[][]{{"right", "CC", "WRB"}});
nonTerminalInfo.put("WHNP", new String[][]{{"left", "WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP"}});
nonTerminalInfo.put("WHPP", new String[][]{{"right", "IN", "TO", "FW"}});
nonTerminalInfo.put("X", new String[][]{{"right"}}); // crap rule
nonTerminalInfo.put("NP", new String[][]{{"rightdis", "NN", "NNP", "NNPS", "NNS", "NX", "POS", "JJR"}, {"left", "NP"}, {"rightdis", "$", "ADJP", "PRN"}, {"right", "CD"}, {"rightdis", "JJ", "JJS", "RB", "QP"}});
nonTerminalInfo.put("TYPO", new String[][] {{"left"}}); // another crap rule, for Brown (Roger)
nonTerminalInfo.put("EDITED", new String[][] {{"left"}}); // crap rule for Switchboard (if don't delete EDITED nodes)
nonTerminalInfo.put("XS", new String[][] {{"right", "IN"}}); // rule for new structure in QP
}

/*

 "left" means search left-to-right by category and then by position
"leftdis" means search left-to-right by position and then by category
"right" means search right-to-left by category and then by position
"rightdis" means search right-to-left by position and then by category
"leftexcept" means to take the first thing from the left that isn't in the list
"rightexcept" means to take the first thing from the right that isn't on the list

Basically, this means that

*/
