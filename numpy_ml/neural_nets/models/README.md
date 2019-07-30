# Models

The models module implements popular full neural networks. It includes:

- `vae.py`: A Bernoulli variational autoencoder ([Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114))
- `wgan_gp.py`: A Wasserstein generative adversarial network with gradient
      penalty ([Gulrajani et al., 2017](https://arxiv.org/pdf/1704.00028.pdf);
[Goodfellow et al., 2014](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf))
- `w2v.py`: word2vec model with CBOW and skip-gram architectures and
  training via noise contrastive estimation ([Mikolov et al., 2012](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf))
