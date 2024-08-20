import nltk
import sys
import re

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP Conj VP
NP -> N | Det N | Det AP N | P NP | NP P NP
VP -> V | Adv VP | V Adv | VP NP | V NP Adv
AP -> Adj | AP Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    s = preprocess(s)

    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    
    if not trees:
        print("Could not parse sentence")
        return

    for t in trees:
        t.pretty_print()
        print("Noun Phrase Chunks")
        for np in np_chunk(t):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    vTest = re.compile('[a-zA-Z]')
    tokens = nltk.word_tokenize(sentence)
    return [entry.lower() for entry in tokens if vTest.match(entry)]


def np_chunk(tree):
    chunks = []
    pTree = nltk.tree.ParentedTree.convert(tree)

    for sTree in pTree.subtrees():
        if sTree.label() == "N":
            chunks.append(sTree.parent())
    return chunks


if __name__ == "__main__":
    main()