from re import split
import nltk
import sys

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
S -> NP VP | NP JP NP | NP VP JP NP
S -> JP NP VP | JP NP VP JP NP | NP VP JP NP
S -> S Conj VP NP | S Conj S | JP S | S Conj VP JP NP
NP -> N | NP Adv | NP JP NP | Adj NP
VP -> V | Adv VP | VP Adv | VP NP
JP -> P | Det | P Det
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def get_token_symbols(token: str):
    return [p.lhs().symbol() for p in grammar.productions(None, token)]


def print_sentence_debug(tokens: list[str]):
    print("Chart:", parser.chart_parse(tokens).pretty_format())
    print("\nText:", *tokens)
    all_symbols = []
    for token in tokens:
        token_symbols = get_token_symbols(token)
        all_symbols += token_symbols
        print(
            *token_symbols,
            " - ",
            token,
        )
    print("\nSymbols:", *all_symbols)


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
        print("Could not parse sentence.")
        return

    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def contains_some_alpha_chars(word: str):
    return any(c.isalpha() for c in word.strip())


def preprocess(sentence: str):
    return [
        word
        for word in nltk.word_tokenize(sentence.lower())
        if contains_some_alpha_chars(word)
    ]


def is_np_tree(tree: nltk.Tree) -> bool:
    return tree.label() == "NP"


def is_np_chunk(tree: nltk.Tree) -> bool:
    if not is_np_tree(tree):
        return False

    for subtree in tree.subtrees():
        if tree == subtree:
            continue

        if is_np_tree(subtree):
            return False
    return True


def np_chunk(tree: nltk.Tree):
    np_chunks = [
        np_subtree
        for np_subtree in tree.subtrees(is_np_tree)
        if is_np_chunk(np_subtree)
    ]
    return np_chunks


if __name__ == "__main__":
    main()