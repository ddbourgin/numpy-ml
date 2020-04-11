# Modules

The `modules.py` module implements common multi-layer blocks that appear across
many modern deep networks. It includes:

- Bidirectional LSTMs ([Schuster & Paliwal, 1997](https://pdfs.semanticscholar.org/4b80/89bc9b49f84de43acc2eb8900035f7d492b2.pdf))
- ResNet-style "identity" (i.e., `same`-convolution) residual blocks ([He et al., 2015](https://arxiv.org/pdf/1512.03385.pdf))
- ResNet-style "convolutional" (i.e., parametric) residual blocks ([He et al., 2015](https://arxiv.org/pdf/1512.03385.pdf))
- WaveNet-style residual block with dilated causal convolutions ([van den Oord et al., 2016](https://arxiv.org/pdf/1609.03499.pdf))
- Transformer-style multi-headed dot-product attention ([Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf))
