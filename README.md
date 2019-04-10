# numpy-ml
Ever wish you had an inefficient but somewhat legible collection of machine
learning algorithms implemented exclusively in numpy? No? Well here you
go anyway.

## Models
This repo includes code for the following models:

1. **Gaussian mixture model**
    - EM training

2. **Hidden Markov model**
    - Viterbi decoding
    - Likelihood computation
    - Baum-Welch/Forward-backward estimation

3. **Latent Dirichlet allocation** (topic model)
    - Standard model with MLE parameter estimation via variational EM
    - Smoothed model with MAP parameter estimation via MCMC 

4. **Neural networks** 
    * Layers / Layer-wise ops
        - Elman-style RNN 
        - LSTM 
        - Fully connected
        - Max + average pooling 
        - Restricted Boltzmann machine (w. CD-n training)
        - 2D deconvolution (w. padding and stride)
        - 2D convolution (w. padding, dilation, and stride)
        - 1D convolution (w. padding, dilation, stride, and causality)
        - Add
        - Multiply
        - Flatten
    * Modules
        - Bidirectional LSTM 
        - ResNet-style residual blocks (identity and convolution)
        - WaveNet-style residual blocks with dilated causal convolutions
    * Regularizers
        - Dropout 
    * Normalization
        - Batch normalization (spatial and temporal)
    * Optimizers
        - SGD w/ momentum 
        - AdaGrad 
        - RMSProp 
    * Initializers
        - Glorot/Xavier uniform and normal
        - He/Kaiming uniform and normal
    * Losses
        - Cross entropy
        - Squared error
        - Bernoulli VAE loss
    * Activations
        - ReLU
        - Tanh
        - Affine
        - Sigmoid
        - Softmax
        - Leaky ReLU
    * Models
        - Bernoulli VAE
    * Utilities
        - `col2im` (MATLAB port)
        - `im2col` (MATLAB port)
        - `conv1D`
        - `conv2D`
        - `deconv2D`

5. **Tree-based models**
    - Decision trees (CART)
    - [Bagging] Random forests 
    - [Boosting] Gradient-boosted decision trees

6. **Linear models**
    - Logistic regression
    - Ordinary least squares 
    - Bayesian linear regression w/ conjugate priors
        - Unknown mean, known variance (Gaussian prior)
        - Unknown mean, unknown variance (Normal-Gamma / Normal-Inverse-Wishart prior)

6. **Preprocessing**
    - Discrete Fourier transform (1D signals)
    - Bilinear interpolation (2D signals)
    - Nearest neighbor interpolation (1D and 2D signals)

## Contributing

Am I missing your favorite model? Is there something that could be cleaner /
less confusing? Did I mess something up? Submit a PR! The only requirement is
that your models are written using just the [Python standard library](https://docs.python.org/3/library/) and numpy.

