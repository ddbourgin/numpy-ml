# Layers
The `layers.py` module implements common layers / layer-wise operations that can
be composed to create larger neural networks. It includes:

- Fully-connected layers
- Sparse evolutionary layers ([Mocanu et al., 2018](https://www.nature.com/articles/s41467-018-04316-3))
- Dot-product attention layers ([Luong, Pho, & Manning, 2015](https://arxiv.org/pdf/1508.04025.pdf); [Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf))
- 1D and 2D convolution (with stride, padding, and dilation) layers ([van den Oord et al., 2016](https://arxiv.org/pdf/1609.03499.pdf); [Yu & Kolton, 2016](https://arxiv.org/pdf/1511.07122.pdf))
- 2D "deconvolution" (with stride and padding) layers ([Zeiler et al., 2010](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf))
- Restricted Boltzmann machines (with CD-_n_ training) ([Smolensky, 1996](http://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf); [Carreira-Perpiñán & Hinton, 2005](http://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf))
- Elementwise multiplication operation
- Summation operation
- Flattening operation
- Embedding layer
- Softmax layer
- Max & average pooling layer
- 1D and 2D batch normalization layers ([Ioffe & Szegedy, 2015](http://proceedings.mlr.press/v37/ioffe15.pdf))
- 1D and 2D layer normalization layers ([Ba, Kiros, & Hinton, 2016](https://arxiv.org/pdf/1607.06450.pdf))
- Recurrent layers ([Elman, 1990](https://crl.ucsd.edu/~elman/Papers/fsit.pdf))
- Long short-term memory (LSTM) layers ([Hochreiter & Schmidhuber, 1997](http://www.bioinf.jku.at/publications/older/2604.pdf))
