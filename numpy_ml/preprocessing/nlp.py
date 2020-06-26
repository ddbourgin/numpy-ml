"""Common preprocessing utilities for working with text data"""
import re
import heapq
import os.path as op
from collections import Counter

import numpy as np


# This list of English stop words is taken from the "Glasgow Information
# Retrieval Group". The original list can be found at
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
_STOP_WORDS = {
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
}

_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

_WORD_REGEX = re.compile(r"(?u)\b\w\w+\b")  # sklearn default
_PUNC_TABLE = str.maketrans("", "", _PUNCTUATION)


def ngrams(sequence, N):
    """Return all `N`-grams of the elements in `sequence`"""
    assert N >= 1
    return list(zip(*[sequence[i:] for i in range(N)]))


def tokenize_words(line, lowercase=True, filter_stopwords=True):
    """
    Split a string into individual lower-case words, optionally removing
    punctuation and stop-words in the process
    """
    words = _WORD_REGEX.findall(line.lower() if lowercase else line)
    return remove_stop_words(words) if filter_stopwords else words


def tokenize_chars(line, lowercase=True, filter_punctuation=True):
    """
    Split a string into individual lower-case words, optionally removing
    punctuation and stop-words in the process
    """
    line = line.lower() if lowercase else line
    line = strip_punctuation(line) if filter_punctuation else line
    chars = list(re.sub(" {2,}", " ", line).strip())
    return chars


def remove_stop_words(words):
    """Remove stop words from a list of word strings"""
    return [w for w in words if w not in _STOP_WORDS]


def strip_punctuation(line):
    """Remove punctuation from a string"""
    return line.translate(_PUNC_TABLE).strip()


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
        """Greater than"""
        if not isinstance(other, Node):
            return -1
        return self.val > other.val

    def __ge__(self, other):
        """Greater than or equal to"""
        if not isinstance(other, Node):
            return -1
        return self.val >= other.val

    def __lt__(self, other):
        """Less than"""
        if not isinstance(other, Node):
            return -1
        return self.val < other.val

    def __le__(self, other):
        """Less than or equal to"""
        if not isinstance(other, Node):
            return -1
        return self.val <= other.val


class HuffmanEncoder(object):
    def fit(self, text):
        """
        Build a Huffman tree for the tokens in `text` and compute each token's
        binary encoding.

        Notes
        -----
        In a Huffman code, tokens that occur more frequently are (generally)
        represented using fewer bits. Huffman codes produce the minimum expected
        codeword length among all methods for encoding tokens individually.

        Huffman codes correspond to paths through a binary tree, with 1
        corresponding to "move right" and 0 corresponding to "move left". In
        contrast to standard binary trees, the Huffman tree is constructed from the
        bottom up. Construction begins by initializing a min-heap priority queue
        consisting of each token in the corpus, with priority corresponding to the
        token frequency. At each step, the two most infrequent tokens in the corpus
        are removed and become the children of a parent pseudotoken whose
        "frequency" is the sum of the frequencies of its children. This new parent
        pseudotoken is added to the priority queue and the process is repeated
        recursively until no tokens remain.

        Parameters
        ----------
        text: list of strs or :class:`Vocabulary` instance
            The tokenized text or a pretrained :class:`Vocabulary` object to use for
            building the Huffman code.
        """
        self._build_tree(text)
        self._generate_codes()

    def transform(self, text):
        """
        Transform the words in `text` into their Huffman-code representations.

        Parameters
        ----------
        text: list of `N` strings
            The list of words to encode

        Returns
        -------
        codes : list of `N` binary strings
            The encoded words in `text`
        """
        if isinstance(text, str):
            text = [text]
        for token in set(text):
            if token not in self._item2code:
                raise Warning("Token '{}' not in Huffman tree. Skipping".format(token))
        return [self._item2code.get(t, None) for t in text]

    def inverse_transform(self, codes):
        """
        Transform an encoded sequence of bit-strings back into words.

        Parameters
        ----------
        codes : list of `N` binary strings
            A list of encoded bit-strings, represented as strings.

        Returns
        -------
        text: list of `N` strings
            The decoded text.
        """
        if isinstance(codes, str):
            codes = [codes]
        for code in set(codes):
            if code not in self._code2item:
                raise Warning("Code '{}' not in Huffman tree. Skipping".format(code))
        return [self._code2item.get(c, None) for c in codes]

    @property
    def tokens(self):
        """A list the unique tokens in `text`"""
        return list(self._item2code.keys())

    @property
    def codes(self):
        """A list with the Huffman code for each unique token in `text`"""
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
        """A string representation of the token"""
        return "Token(word='{}', count={})".format(self.word, self.count)


class TFIDFEncoder:
    def __init__(
        self,
        vocab=None,
        lowercase=True,
        min_count=0,
        smooth_idf=True,
        max_tokens=None,
        input_type="filename",
        filter_stopwords=True,
    ):
        r"""
        An object for compiling and encoding the term-frequency
        inverse-document-frequency (TF-IDF) representation of the tokens in a
        text corpus.

        Notes
        -----
        TF-IDF is intended to reflect how important a word is to a document in
        a collection or corpus. For a word token `w` in a document `d`, and a
        corpus, :math:`D = \{d_1, \ldots, d_N\}`, we have:

        .. math::
            \text{TF}(w, d)  &=  \text{num. occurences of }w \text{ in document }d \\
            \text{IDF}(w, D)  &=  \log \frac{|D|}{|\{ d \in D: t \in d \}|}

        Parameters
        ----------
        vocab : :class:`Vocabulary` object or list-like
            An existing vocabulary to filter the tokens in the corpus against.
            Default is None.
        lowercase : bool
            Whether to convert each string to lowercase before tokenization.
            Default is True.
        min_count : int
            Minimum number of times a token must occur in order to be included
            in vocab. Default is 0.
        smooth_idf : bool
            Whether to add 1 to the denominator of the IDF calculation to avoid
            divide-by-zero errors. Default is True.
        max_tokens : int
            Only add the `max_tokens` most frequent tokens that occur more
            than `min_count` to the vocabulary.  If None, add all tokens
            greater that occur more than than `min_count`. Default is None.
        input_type : {'filename', 'strings'}
            If 'files', the sequence input to `fit` is expected to be a list
            of filepaths. If 'strings', the input is expected to be a list of
            lists, each sublist containing the raw strings for a single
            document in the corpus. Default is 'filename'.
        filter_stopwords : bool
            Whether to remove stopwords before encoding the words in the
            corpus. Default is True.
        """
        # create a function to filter against words in the vocab
        self._filter_vocab = lambda words: words
        if isinstance(vocab, Vocabulary):
            self._filter_vocab = vocab.filter
        elif isinstance(vocab, (list, np.ndarray, set)):
            vocab = set(vocab)
            self._filter_vocab = lambda words: [
                w if w in vocab else "<unk>" for w in words
            ]

        if input_type not in ["files", "strings"]:
            fstr = "`input_type` must be either 'files' or 'strings', but got {}"
            raise ValueError(fstr.format(input_type))

        self._tokens = None
        self._idx2doc = None
        self.term_freq = None
        self.idx2token = None
        self.token2idx = None
        self.inv_doc_freq = None

        self.hyperparameters = {
            "id": "TFIDFEncoder",
            "encoding": None,
            "vocab": vocab
            if not isinstance(vocab, Vocabulary)
            else vocab.hyperparameters,
            "lowercase": lowercase,
            "min_count": min_count,
            "input_type": input_type,
            "max_tokens": max_tokens,
            "smooth_idf": smooth_idf,
            "filter_stopwords": filter_stopwords
            if not isinstance(vocab, Vocabulary)
            else vocab.hyperparameters["filter_stopwords"],
        }

    def fit(self, corpus_seq, encoding="utf-8-sig"):
        """
        Compute term-frequencies and inverse document frequencies on a
        collection of documents.

        Parameters
        ----------
        corpus_seq : str or list of strs
            The filepath / list of filepaths / raw string contents of the
            document(s) to be encoded, in accordance with the `input_type`
            parameter passed to the :meth:`__init__` method. Each document is
            expected to be a newline-separated strings of text, with adjacent
            tokens separated by a whitespace character.
        encoding : str
            Specifies the text encoding for corpus if `input_type` is `files`.
            Common entries are either 'utf-8' (no header byte), or 'utf-8-sig'
            (header byte). Default is 'utf-8-sig'.
        """
        H = self.hyperparameters

        if isinstance(corpus_seq, str):
            corpus_seq = [corpus_seq]

        if H["input_type"] == "files":
            for corpus_fp in corpus_seq:
                assert op.isfile(corpus_fp), "{} does not exist".format(corpus_fp)

        tokens = []
        idx2token, token2idx = {}, {}

        # encode special tokens
        for tt in ["<bol>", "<eol>", "<unk>"]:
            token2idx[tt] = len(tokens)
            idx2token[len(tokens)] = tt
            tokens.append(Token(tt))

        min_count = H["min_count"]
        max_tokens = H["max_tokens"]
        H["encoding"] = encoding

        bol_ix = token2idx["<bol>"]
        eol_ix = token2idx["<eol>"]
        idx2doc, term_freq = {}, {}

        # encode the text in `corpus_fps` without any filtering ...
        for d_ix, doc in enumerate(corpus_seq):
            doc_count = {}
            idx2doc[d_ix] = doc if H["input_type"] == "files" else None
            token2idx, idx2token, tokens, doc_count = self._encode_document(
                doc, token2idx, idx2token, tokens, doc_count, bol_ix, eol_ix,
            )
            term_freq[d_ix] = doc_count

        self._tokens = tokens
        self._idx2doc = idx2doc
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.term_freq = term_freq

        # ... retain only the top `max_tokens` most frequent tokens, coding
        # everything else as <unk> ...
        if max_tokens is not None and len(tokens) > max_tokens:
            self._keep_top_n_tokens()

        # ... replace all words occurring less than `min_count` by <unk> ...
        if min(self._tokens, key=lambda t: t.count).count < min_count:
            self._drop_low_freq_tokens()

        # ... sort tokens alphabetically and reindex ...
        self._sort_tokens()

        # ... finally, calculate inverse document frequency
        self._calc_idf()

    def _encode_document(
        self, doc, word2idx, idx2word, tokens, doc_count, bol_ix, eol_ix,
    ):
        """Perform tokenization and compute token counts for a single document"""
        H = self.hyperparameters
        lowercase = H["lowercase"]
        filter_stop = H["filter_stopwords"]

        if H["input_type"] == "files":
            with open(doc, "r", encoding=H["encoding"]) as handle:
                doc = handle.read()

        n_words = 0
        lines = doc.split("\n")
        for line in lines:
            words = tokenize_words(line, lowercase, filter_stop)
            words = self._filter_vocab(words)
            n_words += len(words)

            for ww in words:
                if ww not in word2idx:
                    word2idx[ww] = len(tokens)
                    idx2word[len(tokens)] = ww
                    tokens.append(Token(ww))

                t_idx = word2idx[ww]
                tokens[t_idx].count += 1
                doc_count[t_idx] = doc_count.get(t_idx, 0) + 1

            # wrap line in <bol> and <eol> tags
            tokens[bol_ix].count += 1
            tokens[eol_ix].count += 1

            doc_count[bol_ix] = doc_count.get(bol_ix, 0) + 1
            doc_count[eol_ix] = doc_count.get(eol_ix, 0) + 1
        return word2idx, idx2word, tokens, doc_count

    def _keep_top_n_tokens(self):
        N = self.hyperparameters["max_tokens"]
        doc_counts, word2idx, idx2word = {}, {}, {}
        tokens = sorted(self._tokens, key=lambda x: x.count, reverse=True)

        # reindex the top-N tokens...
        unk_ix = None
        for idx, tt in enumerate(tokens[:N]):
            word2idx[tt.word] = idx
            idx2word[idx] = tt.word

            if tt.word == "<unk>":
                unk_ix = idx

        # ... if <unk> isn't in the top-N, add it, replacing the Nth
        # most-frequent word and adjust the <unk> count accordingly ...
        if unk_ix is None:
            unk_ix = self.token2idx["<unk>"]
            old_count = tokens[N - 1].count
            tokens[N - 1] = self._tokens[unk_ix]
            tokens[N - 1].count += old_count
            word2idx["<unk>"] = N - 1
            idx2word[N - 1] = "<unk>"

        # ... and recode all dropped tokens as "<unk>"
        for tt in tokens[N:]:
            tokens[unk_ix].count += tt.count

        # ... finally, reindex the word counts for each document
        doc_counts = {}
        for d_ix in self.term_freq.keys():
            doc_counts[d_ix] = {}
            for old_ix, d_count in self.term_freq[d_ix].items():
                word = self.idx2token[old_ix]
                new_ix = word2idx.get(word, unk_ix)
                doc_counts[d_ix][new_ix] = doc_counts[d_ix].get(new_ix, 0) + d_count

        self._tokens = tokens[:N]
        self.token2idx = word2idx
        self.idx2token = idx2word
        self.term_freq = doc_counts

        assert len(self._tokens) <= N

    def _drop_low_freq_tokens(self):
        """
        Replace all tokens that occur less than `min_count` with the `<unk>`
        token.
        """
        H = self.hyperparameters
        unk_token = self._tokens[self.token2idx["<unk>"]]
        eol_token = self._tokens[self.token2idx["<eol>"]]
        bol_token = self._tokens[self.token2idx["<bol>"]]
        tokens = [unk_token, eol_token, bol_token]

        unk_idx = 0
        word2idx = {"<unk>": 0, "<eol>": 1, "<bol>": 2}
        idx2word = {0: "<unk>", 1: "<eol>", 2: "<bol>"}
        special = {"<eol>", "<bol>", "<unk>"}

        for tt in self._tokens:
            if tt.word not in special:
                if tt.count < H["min_count"]:
                    tokens[unk_idx].count += tt.count
                else:
                    word2idx[tt.word] = len(tokens)
                    idx2word[len(tokens)] = tt.word
                    tokens.append(tt)

        # reindex document counts
        doc_counts = {}
        for d_idx in self.term_freq.keys():
            doc_counts[d_idx] = {}
            for old_idx, d_count in self.term_freq[d_idx].items():
                word = self.idx2token[old_idx]
                new_idx = word2idx.get(word, unk_idx)
                doc_counts[d_idx][new_idx] = doc_counts[d_idx].get(new_idx, 0) + d_count

        self._tokens = tokens
        self.token2idx = word2idx
        self.idx2token = idx2word
        self.term_freq = doc_counts

    def _sort_tokens(self):
        # sort tokens alphabetically and recode
        ix = 0
        token2idx, idx2token, = {}, {}
        special = ["<eol>", "<bol>", "<unk>"]
        words = sorted(self.token2idx.keys())
        term_freq = {d: {} for d in self.term_freq.keys()}

        for w in words:
            if w not in special:
                old_ix = self.token2idx[w]
                token2idx[w], idx2token[ix] = ix, w
                for d in self.term_freq.keys():
                    if old_ix in self.term_freq[d]:
                        count = self.term_freq[d][old_ix]
                        term_freq[d][ix] = count
                ix += 1

        for w in special:
            token2idx[w] = len(token2idx)
            idx2token[len(idx2token)] = w

        self.token2idx = token2idx
        self.idx2token = idx2token
        self.term_freq = term_freq
        self.vocab_counts = Counter({t.word: t.count for t in self._tokens})

    def _calc_idf(self):
        """
        Compute the (smoothed-) inverse-document frequency for each token in
        the corpus.

        For a word token `w`, the IDF is simply

            IDF(w) = log ( |D| / |{ d in D: w in d }| ) + 1

        where D is the set of all documents in the corpus,

            D = {d1, d2, ..., dD}

        If `smooth_idf` is True, we perform additive smoothing on the number of
        documents containing a given word, equivalent to pretending that there
        exists a final D+1st document that contains every word in the corpus:

            SmoothedIDF(w) = log ( |D| + 1 / [1 + |{ d in D: w in d }|] ) + 1
        """
        inv_doc_freq = {}
        smooth_idf = self.hyperparameters["smooth_idf"]
        tf, doc_idxs = self.term_freq, self._idx2doc.keys()

        D = len(self._idx2doc) + int(smooth_idf)
        for word, w_ix in self.token2idx.items():
            d_count = int(smooth_idf)
            d_count += np.sum([1 if w_ix in tf[d_ix] else 0 for d_ix in doc_idxs])
            inv_doc_freq[w_ix] = 1 if d_count == 0 else np.log(D / d_count) + 1
        self.inv_doc_freq = inv_doc_freq

    def transform(self, ignore_special_chars=True):
        """
        Generate the term-frequency inverse-document-frequency encoding of a
        text corpus.

        Parameters
        ----------
        ignore_special_chars : bool
            Whether to drop columns corresponding to "<eol>", "<bol>", and
            "<unk>" tokens from the final tfidf encoding. Default is True.

        Returns
        -------
        tfidf : numpy array of shape `(D, M [- 3])`
            The encoded corpus, with each row corresponding to a single
            document, and each column corresponding to a token id. The mapping
            between column numbers and tokens is stored in the `idx2token`
            attribute IFF `ignore_special_chars` is False. Otherwise, the
            mappings are not accurate.
        """
        D, N = len(self._idx2doc), len(self._tokens)
        tf = np.zeros((D, N))
        idf = np.zeros((D, N))

        for d_ix in self._idx2doc.keys():
            words, counts = zip(*self.term_freq[d_ix].items())
            docs = np.ones(len(words), dtype=int) * d_ix
            tf[docs, words] = counts

        words = sorted(self.idx2token.keys())
        idf = np.tile(np.array([self.inv_doc_freq[w] for w in words]), (D, 1))
        tfidf = tf * idf

        if ignore_special_chars:
            idxs = [
                self.token2idx["<unk>"],
                self.token2idx["<eol>"],
                self.token2idx["<bol>"],
            ]
            tfidf = np.delete(tfidf, idxs, 1)

        return tfidf


class Vocabulary:
    def __init__(
        self, lowercase=True, min_count=None, max_tokens=None, filter_stopwords=True,
    ):
        """
        An object for compiling and encoding the unique tokens in a text corpus.

        Parameters
        ----------
        lowercase : bool
            Whether to convert each string to lowercase before tokenization.
            Default is True.
        min_count : int
            Minimum number of times a token must occur in order to be included
            in vocab. If `None`, include all tokens from `corpus_fp` in vocab.
            Default is None.
        max_tokens : int
            Only add the `max_tokens` most frequent tokens that occur more
            than `min_count` to the vocabulary.  If None, add all tokens
            greater that occur more than than `min_count`. Default is None.
        filter_stopwords : bool
            Whether to remove stopwords before encoding the words in the
            corpus. Default is True.
        """
        self.hyperparameters = {
            "id": "Vocabulary",
            "encoding": None,
            "corpus_fps": None,
            "lowercase": lowercase,
            "min_count": min_count,
            "max_tokens": max_tokens,
            "filter_stopwords": filter_stopwords,
        }

    def __len__(self):
        """Return the number of tokens in the vocabulary"""
        return len(self._tokens)

    def __iter__(self):
        """Return an iterator over the tokens in the vocabulary"""
        return iter(self._tokens)

    def __contains__(self, word):
        """Assert whether `word` is a token in the vocabulary"""
        return word in self.token2idx

    def __getitem__(self, key):
        """
        Return the token (if key is an integer) or the index (if key is a string)
        for the key in the vocabulary, if it exists.
        """
        if isinstance(key, str):
            return self._tokens[self.token2idx[key]]
        if isinstance(key, int):
            return self._tokens[key]

    @property
    def n_tokens(self):
        """The number of unique word tokens in the vocabulary"""
        return len(self.token2idx)

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

    def filter(self, words, unk=True):  # noqa: A003
        """
        Filter or replace any word in `words` that does not occur in
        `Vocabulary`

        Parameters
        ----------
        words : list of strs
            A list of words to filter
        unk : bool
            Whether to replace any out of vocabulary words in `words` with the
            <unk> token (unk = True) or skip them entirely (unk = False).
            Default is True.

        Returns
        -------
        filtered : list of strs
            The list of words filtered against the vocabulary.
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
        unk_ix = self.token2idx["<unk>"]
        lowercase = self.hyperparameters["lowercase"]
        words = [w.lower() for w in words] if lowercase else words
        return [self.token2idx[w] if w in self else unk_ix for w in words]

    def indices_to_words(self, indices):
        """
        Convert the indices in `indices` to their word values. If an index is
        not in the vocabulary, return the the <unk> token.

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
        return [self.idx2token[i] if i in self.idx2token else unk for i in indices]

    def fit(self, corpus_fps, encoding="utf-8-sig"):
        """
        Compute the vocabulary across a collection of documents.

        Parameters
        ----------
        corpus_fps : str or list of strs
            The filepath / list of filepaths for the document(s) to be encoded.
            Each document is expected to be encoded as newline-separated
            string of text, with adjacent tokens separated by a whitespace
            character.
        encoding : str
            Specifies the text encoding for corpus. Common entries are either
            'utf-8' (no header byte), or 'utf-8-sig' (header byte). Default is
            'utf-8-sig'.
        """
        if isinstance(corpus_fps, str):
            corpus_fps = [corpus_fps]

        for corpus_fp in corpus_fps:
            assert op.isfile(corpus_fp), "{} does not exist".format(corpus_fp)

        tokens = []
        H = self.hyperparameters
        idx2word, word2idx = {}, {}

        min_count = H["min_count"]
        lowercase = H["lowercase"]
        max_tokens = H["max_tokens"]
        filter_stop = H["filter_stopwords"]

        H["encoding"] = encoding
        H["corpus_fps"] = corpus_fps

        # encode special tokens
        for tt in ["<bol>", "<eol>", "<unk>"]:
            word2idx[tt] = len(tokens)
            idx2word[len(tokens)] = tt
            tokens.append(Token(tt))

        bol_ix = word2idx["<bol>"]
        eol_ix = word2idx["<eol>"]

        for d_ix, doc_fp in enumerate(corpus_fps):
            with open(doc_fp, "r", encoding=H["encoding"]) as doc:
                for line in doc:
                    words = tokenize_words(line, lowercase, filter_stop)

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
        self.token2idx = word2idx
        self.idx2token = idx2word

        # replace all words occurring less than `min_count` by <unk>
        if min_count is not None:
            self._drop_low_freq_tokens()

        # retain only the top `max_tokens` most frequent tokens, coding
        # everything else as <unk>
        if max_tokens is not None and len(tokens) > max_tokens:
            self._keep_top_n_tokens()

        counts = {w: self._tokens[ix].count for w, ix in self.token2idx.items()}
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
            unk_ix = self.token2idx["<unk>"]
            old_count = tokens[N - 1].count
            tokens[N - 1] = self._tokens[unk_ix]
            tokens[N - 1].count += old_count
            word2idx["<unk>"] = N - 1
            idx2word[N - 1] = "<unk>"

        # ... and recode all dropped tokens as "<unk>"
        for tt in tokens[N:]:
            tokens[unk_ix].count += tt.count

        self._tokens = tokens[:N]
        self.token2idx = word2idx
        self.idx2token = idx2word

        assert len(self._tokens) <= N

    def _drop_low_freq_tokens(self):
        """
        Replace all tokens that occur less than `min_count` with the `<unk>`
        token.
        """
        unk_idx = 0
        unk_token = self._tokens[self.token2idx["<unk>"]]
        eol_token = self._tokens[self.token2idx["<eol>"]]
        bol_token = self._tokens[self.token2idx["<bol>"]]

        H = self.hyperparameters
        tokens = [unk_token, eol_token, bol_token]
        word2idx = {"<unk>": 0, "<eol>": 1, "<bol>": 2}
        idx2word = {0: "<unk>", 1: "<eol>", 2: "<bol>"}
        special = {"<eol>", "<bol>", "<unk>"}

        for tt in self._tokens:
            if tt.word not in special:
                if tt.count < H["min_count"]:
                    tokens[unk_idx].count += tt.count
                else:
                    word2idx[tt.word] = len(tokens)
                    idx2word[len(tokens)] = tt.word
                    tokens.append(tt)

        self._tokens = tokens
        self.token2idx = word2idx
        self.idx2token = idx2word
