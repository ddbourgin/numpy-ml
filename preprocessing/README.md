# Preprocessing
The preprocessing module implements common data preprocessing routines.

- `nlp.py`: Routines and objects for handling text data.
    - n-gram generators
    - Word and character tokenization
    - Punctuation and stop-word removal
    - Vocabulary / unigram count objects
    - [Huffman tree](https://en.wikipedia.org/wiki/Huffman_coding) encoding / decoding
    - Term frequency-inverse document frequency ([tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) encoding

- `dsp.py`: Routines for handling audio and image data.
    - Signal windowing
    - Signal autocorrelation
    - Discrete Fourier transform    
    - Signal resampling via (bi-)linear interpolation and nearest neighbor
    
- `general.py`: General data preprocessing objects and functions.
    - Feature hashing ([Moody, 1989](http://papers.nips.cc/paper/175-fast-learning-in-multi-resolution-hierarchies.pdf))
    - Mini-batch generators
    - One-hot encoding / decoding
    - Feature standardization
