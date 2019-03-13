# numpy-ml

Ever wish you had an inefficient but somewhat legible collection of machine
learning algorithms written exclusively in numpy? No? Well here it is anyway.

## Models

1. Gaussian mixture model
    - EM training

2. Hidden Markov model
    - Viterbi decoding
    - Likelihood computation
    - Baum-Welch/Forward-backward estimation

3. Latent Dirichlet allocation
    - Vanilla model with variational EM training
    - Smoothed model with MCMC (collapsed Gibbs sampler) training

4. Neural networks (Keras-style)
    - Elman-style RNN layer + cell
    - LSTM layer + cell
    - 2D convolutional layer
    - Fully-connected layer
    - Max + average pooling layers
    - Bidirectional LSTM module
    - ResNet-style identity and convolution modules
    - Dropout regularization 
    - Batch normalization (spatial and temporal)
    - SGD w/ momentum 
    - AdaGrad 
    - RMSProp 

5. Data preprocessing
    - Image resampling via bilinear interpolation

## Requirements

Freaking numpy. Also scipy if you *really* want to use the variational EM LDA
model.
