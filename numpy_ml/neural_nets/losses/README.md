# Losses

The `losses.py` module implements several common loss functions, including:

- Squared error
- Cross-entropy
- Variational lower-bound for binary VAE ([Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114))
- WGAN-GP loss for generator and critic ([Gulrajani et al., 2017](https://arxiv.org/pdf/1704.00028.pdf))
- Noise contrastive estimation (NCE) loss ([Gutmann &
  Hyv&auml;rinen, 2010](https://www.cs.helsinki.fi/u/ahyvarin/papers/Gutmann10AISTATS.pdf); [Minh & Teh, 2012](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf))
