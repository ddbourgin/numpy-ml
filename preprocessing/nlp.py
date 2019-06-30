import re
import heapq
import os.path as op
from collections import Counter

import numpy as np


# This list of English stop words is taken from the "Glasgow Information
# Retrieval Group". The original list can be found at
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
_STOP_WORDS = set(
    [
        "a",
        "about",
        "above",
        "across",
        "after",
        "afterwards",
        "again",
        "against",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "amoungst",
        "amount",
        "an",
        "and",
        "another",
        "any",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "are",
        "around",
        "as",
        "at",
        "back",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "bill",
        "both",
        "bottom",
        "but",
        "by",
        "call",
        "can",
        "cannot",
        "cant",
        "co",
        "con",
        "could",
        "couldnt",
        "cry",
        "de",
        "describe",
        "detail",
        "do",
        "done",
        "down",
        "due",
        "during",
        "each",
        "eg",
        "eight",
        "either",
        "eleven",
        "else",
        "elsewhere",
        "empty",
        "enough",
        "etc",
        "even",
        "ever",
        "every",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "few",
        "fifteen",
        "fifty",
        "fill",
        "find",
        "fire",
        "first",
        "five",
        "for",
        "former",
        "formerly",
        "forty",
        "found",
        "four",
        "from",
        "front",
        "full",
        "further",
        "get",
        "give",
        "go",
        "had",
        "has",
        "hasnt",
        "have",
        "he",
        "hence",
        "her",
        "here",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "hundred",
        "i",
        "ie",
        "if",
        "in",
        "inc",
        "indeed",
        "interest",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "keep",
        "last",
        "latter",
        "latterly",
        "least",
        "less",
        "ltd",
        "made",
        "many",
        "may",
        "me",
        "meanwhile",
        "might",
        "mill",
        "mine",
        "more",
        "moreover",
        "most",
        "mostly",
        "move",
        "much",
        "must",
        "my",
        "myself",
        "name",
        "namely",
        "neither",
        "never",
        "nevertheless",
        "next",
        "nine",
        "no",
        "nobody",
        "none",
        "noone",
        "nor",
        "not",
        "nothing",
        "now",
        "nowhere",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "part",
        "per",
        "perhaps",
        "please",
        "put",
        "rather",
        "re",
        "same",
        "see",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "serious",
        "several",
        "she",
        "should",
        "show",
        "side",
        "since",
        "sincere",
        "six",
        "sixty",
        "so",
        "some",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhere",
        "still",
        "such",
        "system",
        "take",
        "ten",
        "than",
        "that",
        "the",
        "their",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "thereupon",
        "these",
        "they",
        "thick",
        "thin",
        "third",
        "this",
        "those",
        "though",
        "three",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "top",
        "toward",
        "towards",
        "twelve",
        "twenty",
        "two",
        "un",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "via",
        "was",
        "we",
        "well",
        "were",
        "what",
        "whatever",
        "when",
        "whence",
        "whenever",
        "where",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "yet",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)
_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def ngrams(sequence, N):
    """Return all `N`-grams of the elements in `sequence`"""
    assert N >= 1
    return list(zip(*[sequence[i:] for i in range(N)]))


def tokenize_words(line, filter_punctuation=True, filter_stopwords=True):
    """
    Split a string into individual lower-case words, optionally removing
    punctuation and stop-words in the process
    """
    line = strip_punctuation(line) if filter_punctuation else line
    words = line.lower().split()
    return remove_stop_words(words) if filter_stopwords else words


def tokenize_chars(line, filter_punctuation=True):
    """
    Split a string into individual lower-case words, optionally removing
    punctuation and stop-words in the process
    """
    line = strip_punctuation(line) if filter_punctuation else line
    chars = list(re.sub(" {2,}", " ", line.lower()).strip())
    return chars


def remove_stop_words(words):
    """Remove stop words from a list of word strings"""
    return [w for w in words if w not in _STOP_WORDS]


def strip_punctuation(line):
    """Remove punctuation from a string"""
    trans_table = str.maketrans("", "", _PUNCTUATION)
    return line.translate(trans_table).strip()


#######################################################################
#                            Huffman Tree                             #
#######################################################################


class Node(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

    def __gt__(self, other):
        if not isinstance(other, Node):
            return -1
        return self.val > other.val

    def __ge__(self, other):
        if not isinstance(other, Node):
            return -1
        return self.val >= other.val

    def __lt__(self, other):
        if not isinstance(other, Node):
            return -1
        return self.val < other.val

    def __le__(self, other):
        if not isinstance(other, Node):
            return -1
        return self.val <= other.val


class HuffmanEncoder(object):
    """Encode text into a variable-length bit string using a Huffman tree"""

    def fit(self, text):
        """
        Build a Huffman tree for the tokens in `text` and compute each token's
        binary encoding.

        Parameters
        ----------
        text: list of strs or `Vocabulary` instance
            The tokenized text to encode or a pretrained Vocabulary object
        """
        self._build_tree(text)
        self._generate_codes()

    def transform(self, text):
        if isinstance(text, str):
            text = [text]
        for token in set(text):
            if token not in self._item2code:
                raise Warning("Token '{}' not in Huffman tree. Skipping".format(token))
        return [self._item2code.get(t, None) for t in text]

    def inverse_transform(self, codes):
        if isinstance(codes, str):
            codes = [codes]
        for code in set(codes):
            if code not in self._code2item:
                raise Warning("Code '{}' not in Huffman tree. Skipping".format(code))
        return [self._code2item.get(c, None) for c in codes]

    @property
    def tokens(self):
        return list(self._item2code.keys())

    @property
    def codes(self):
        return list(self._code2item.keys())

    def _counter(self, text):
        counts = {}
        for item in text:
            counts[item] = counts.get(item, 0) + 1
        return counts

    def _build_tree(self, text):
        """Construct Huffman Tree"""
        PQ = []

        if isinstance(text, Vocabulary):
            counts = text.counts
        else:
            counts = self._counter(text)

        for (k, c) in counts.items():
            PQ.append(Node(k, c))

        # create a priority queue with priority = item frequency
        heapq.heapify(PQ)

        while len(PQ) > 1:
            node1 = heapq.heappop(PQ)  # item with smallest frequency
            node2 = heapq.heappop(PQ)  # item with second smallest frequency

            parent = Node(None, node1.val + node2.val)
            parent.left = node1
            parent.right = node2

            heapq.heappush(PQ, parent)

        self._root = heapq.heappop(PQ)

    def _generate_codes(self):
        current_code = ""
        self._item2code = {}
        self._code2item = {}
        self._build_code(self._root, current_code)

    def _build_code(self, root, current_code):
        if root is None:
            return

        if root.key is not None:
            self._item2code[root.key] = current_code
            self._code2item[current_code] = root.key
            return

        # 0 = move left, 1 = move right
        self._build_code(root.left, current_code + "0")
        self._build_code(root.right, current_code + "1")


#######################################################################
#                             Containers                              #
#######################################################################


class Token:
    def __init__(self, word):
        self.count = 0
        self.word = word

    def __repr__(self):
        return "Token(word='{}', count={})".format(self.word, self.count)


class Vocabulary:
    def __init__(
        self,
        corpus_fp,
        min_count=None,
        max_tokens=None,
        encoding="utf-8-sig",
        filter_stopwords=True,
        filter_punctuation=True,
    ):
        """
        An object for compiling and encoding the unique tokens in a text corpus.

        Parameters
        ----------
        corpus_fp : str
            The filepath to the text to be encoded. The corpus is expected to
            be encoded as newline-separated strings of text, with adjacent
            tokens separated by a whitespace character.
        min_count : int (default: None)
            Minimum number of times a token must occur in order to be included
            in vocab. If `None`, include all tokens from `corpus_fp` in vocab.
        max_tokens : int (default: None)
            Only add the `max_tokens` most frequent tokens that occur more
            than `min_count` to the vocabulary.  If None, add all tokens
            greater that occur more than than `min_count`.
        encoding : str (default: 'utf-8-sig')
            Specifies the text encoding for corpus. Common entries are either
            'utf-8' (no header byte), or 'utf-8-sig' (header byte).
        filter_stopwords : bool (default: True)
            Whether to remove stopwords before encoding the words in the corpus
        filter_punctuation : bool (default: True)
            Whether to remove punctuation before encoding the words in the
            corpus
        """
        assert op.isfile(corpus_fp), "{} does not exist".format(corpus_fp)

        self.hyperparameters = {
            "id": "Vocabulary",
            "encoding": encoding,
            "corpus_fp": corpus_fp,
            "min_count": min_count,
            "max_tokens": max_tokens,
            "filter_stopwords": filter_stopwords,
            "filter_punctuation": filter_punctuation,
        }

        self._encode()

    def __len__(self):
        return len(self._tokens)

    def __iter__(self):
        return iter(self._tokens.word)

    def __contains__(self, word):
        return word in self._word2idx

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._tokens[self._word2idx[key]]
        if isinstance(key, int):
            return self._tokens[key]

    @property
    def n_tokens(self):
        """The number of unique word tokens in the vocabulary"""
        return len(self._word2idx)

    @property
    def n_words(self):
        """The total number of words in the corpus"""
        return sum(self.counts.values())

    @property
    def shape(self):
        """The number of unique word tokens in the vocabulary"""
        return self._tokens.shape

    def most_common(self, n=5):
        """Return the top `n` most common tokens in the corpus"""
        return self.counts.most_common()[:n]

    def words_with_count(self, k):
        """Return all tokens that occur `k` times in the corpus"""
        return [w for w, c in self.counts.items() if c == k]

    def filter(self, words, unk=True):
        """
        Filter or replace any word in `words` that does not occur in `Vocabulary`

        Parameters
        ----------
        words : list of strs
            A list of words to filter
        unk : bool (default: True)
            Whether to replace any out of vocabulary words in `words` with the
            <unk> token (unk = True) or skip them entirely (unk = False)

        Returns
        -------
        filtered : list of strs
            The list of words filtered against the vocabulary
        """
        if unk:
            return [w if w in self else "<unk>" for w in words]
        return [w for w in words if w in self]

    def words_to_indices(self, words):
        """
        Convert the words in `words` to their token indices. If a word is not
        in the vocabulary, return the index for the <unk> token

        Parameters
        ----------
        words : list of strs
            A list of words to filter

        Returns
        -------
        indices : list of ints
            The token indices for each word in `words`
        """
        unk_ix = self._word2idx["<unk>"]
        words = [w.lower() for w in words]
        return [self._word2idx[w] if w in self else unk_ix for w in words]

    def indices_to_words(self, indices):
        """
        Convert the indices in `indices` to their word values. If an index is not
        in the vocabulary, return the the <unk> token

        Parameters
        ----------
        indices : list of ints
            The token indices for each word in `words`

        Returns
        -------
        words : list of strs
            The word strings corresponding to each token index in `indices`
        """
        unk = "<unk>"
        return [self._idx2word[i] if i in self._idx2word else unk for i in indices]

    def _encode(self):
        tokens = []
        H = self.hyperparameters
        idx2word, word2idx = {}, {}

        filter_stop = H["filter_stopwords"]
        filter_punc = H["filter_punctuation"]

        max_tokens = H["max_tokens"]
        corpus_fp, min_count = H["corpus_fp"], H["min_count"]

        # encode special tokens
        for tt in ["<bol>", "<eol>", "<unk>"]:
            word2idx[tt] = len(tokens)
            idx2word[len(tokens)] = tt
            tokens.append(Token(tt))

        bol_ix = word2idx["<bol>"]
        eol_ix = word2idx["<eol>"]

        with open(corpus_fp, "r", encoding=H["encoding"]) as text:
            for line in text:
                words = tokenize_words(line, filter_punc, filter_stop)

                for ww in words:
                    if ww not in word2idx:
                        word2idx[ww] = len(tokens)
                        idx2word[len(tokens)] = ww
                        tokens.append(Token(ww))

                    t_idx = word2idx[ww]
                    tokens[t_idx].count += 1

                # wrap line in <bol> and <eol> tags
                tokens[bol_ix].count += 1
                tokens[eol_ix].count += 1

        self._tokens = tokens
        self._word2idx = word2idx
        self._idx2word = idx2word

        # replace all words occurring less than `min_count` by <unk>
        if min_count is not None:
            self._drop_low_freq_tokens()

        # retain only the top `max_tokens` most frequent tokens, coding
        # everything else as <unk>
        if max_tokens is not None and len(tokens) > max_tokens:
            self._keep_top_n_tokens()

        counts = {w: self._tokens[ix].count for w, ix in self._word2idx.items()}
        self.counts = Counter(counts)
        self._tokens = np.array(self._tokens)

    def _keep_top_n_tokens(self):
        word2idx, idx2word = {}, {}
        N = self.hyperparameters["max_tokens"]
        tokens = sorted(self._tokens, key=lambda x: x.count, reverse=True)

        # reindex the top-N tokens...
        unk_ix = None
        for idx, tt in enumerate(tokens[:N]):
            word2idx[tt.word] = idx
            idx2word[idx] = tt.word

            if tt.word == "<unk>":
                unk_ix = idx

        # ... if <unk> isn't in the top-N, add it, replacing the Nth
        # most-frequent word and adjusting the <unk> count accordingly ...
        if unk_ix is None:
            unk_ix = self._word2idx["<unk>"]
            old_count = tokens[N - 1].count
            tokens[N - 1] = self._tokens[unk_ix]
            tokens[N - 1].count += old_count
            unk_ix = N - 1

        # ... and recode all dropped tokens as "<unk>"
        for tt in tokens[N:]:
            tokens[unk_ix].count += tt.count

        self._tokens = tokens[:N]
        self._word2idx = word2idx
        self._idx2word = idx2word

        assert len(self._tokens) <= N

    def _drop_low_freq_tokens(self):
        """
        Replace all tokens that occur less than `min_count` with the `<unk>` token.
        """
        unk_idx = 0
        H = self.hyperparameters
        tokens = [Token("<unk>")]
        word2idx, idx2word = {}, {}

        for tt in self._tokens:
            if tt.count < H["min_count"]:
                tokens[unk_idx].count += tt.count
            else:
                word2idx[tt.word] = len(tokens)
                idx2word[len(tokens)] = tt.word
                tokens.append(tt)

        self._tokens = tokens
        self._word2idx = word2idx
        self._idx2word = idx2word
