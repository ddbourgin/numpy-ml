import re
import os.path as op

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
        encoding=None,
        min_count=None,
        max_tokens=None,
        filter_stopwords=True,
        filter_punctuation=True,
    ):
        """
        An object for compiling and encoding the unique tokens in a text corpus.

        Parameters
        ----------
        corpus_fp : str
            The filepath to the text to be encoded
        encoding : str (default: None)
            Specifies the text encoding for corpus. Common entries are either
            'utf-8' (no header byte), or 'utf-8-sig' (header byte).
        min_count : int (default: None)
            Minimum number of times a token must occur in order to be included
            in vocab. If `None`, include all tokens from `corpus_fp` in vocab.
        max_tokens : int (default: None)
            Only add the `max_tokens` most frequent tokens that occur more
            than `min_count` to the vocabulary.  If None, add all tokens
            greater that occur more than than `min_count`.
        filter_stopwords : bool (default: True)
            Whether to remove stopwords before encoding the words in the corpus
        filter_punctuation : bool (default: True)
            Whether to remove punctuation before encoding the words in the
            corpus
        """
        assert op.isfile(corpus_fp), "{} does not exist".format(corpus_fp)

        self.hyperparameters = {
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
        return iter(self._tokens)

    def __contains__(self, word):
        return word in self._word2idx

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._tokens[self._word2idx[key]]
        if isinstance(key, int):
            return self._tokens[key]

    @property
    def words(self):
        """A list of all the words in the vocabulary"""
        return list(self._word2idx.keys())

    @property
    def shape(self):
        """The number of word tokens in the vocabulary"""
        return self._tokens.shape

    def indices(self):
        """Return all valid token indices"""
        return list(self._idx2word.keys())

    def filter_words(self, words, unk=True):
        """
        Filter out or replace any word in `words` that does not occur in
        vocabulary

        Parameters
        ----------
        words : list of strs
            A list of words to filter
        unk : bool (default: True)
            Whether to either replace any out of vocabulary words in `words`
            with the <unk> token or remove them entirely

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

    def unigram_counts(self, words):
        """
        Return the unigram counts in Vocabulary for each word in `words`. If a
        word isn't in the vocabulary, return -1

        Parameters
        ----------
        words : list of strs
            A list of words to count

        Returns
        -------
        counts : list of ints
            The number of occurrences of each word in the corpus
        """
        counts = []
        for word in words:
            count = -1
            word = word.lower()
            if word in self:
                ix = self._word2idx[word]
                count = self._tokens[ix].count
            counts.append(count)
        return counts

    def _encode(self):
        tokens = []
        H = self.hyperparameters
        idx2word, word2idx = {}, {}

        filter_stop = H["filter_stopwords"]
        filter_punc = H["filter_punctuation"]

        max_tokens = H["max_tokens"]
        corpus_fp, min_count = H["corpus_fp"], H["min_count"]

        # encode special characters
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

        assert len(self._tokens) == N

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
