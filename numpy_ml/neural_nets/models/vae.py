from time import time
from collections import OrderedDict

import numpy as np

from ..losses import VAELoss
from ..utils import minibatch
from ..activations import ReLU, Affine, Sigmoid
from ..layers import Conv2D, Pool2D, Flatten, FullyConnected


class BernoulliVAE(object):
    def __init__(
        self,
        T=5,
        latent_dim=256,
        enc_conv1_pad=0,
        enc_conv2_pad=0,
        enc_conv1_out_ch=32,
        enc_conv2_out_ch=64,
        enc_conv1_stride=1,
        enc_pool1_stride=2,
        enc_conv2_stride=1,
        enc_pool2_stride=1,
        enc_conv1_kernel_shape=(5, 5),
        enc_pool1_kernel_shape=(2, 2),
        enc_conv2_kernel_shape=(5, 5),
        enc_pool2_kernel_shape=(2, 2),
        optimizer="RMSProp(lr=0.0001)",
        init="glorot_uniform",
    ):
        """
        A variational autoencoder (VAE) with 2D convolutional encoder and Bernoulli
        input and output units.

        Notes
        -----
        The VAE architecture is

        .. code-block:: text

                            |-- t_mean ----|
            X -> [Encoder] -|              |--> [Sampler] -> [Decoder] -> X_recon
                            |-- t_log_var -|

        where ``[Encoder]`` is

        .. code-block:: text

            Conv1 -> ReLU -> MaxPool1 -> Conv2 -> ReLU ->
                MaxPool2 -> Flatten -> FC1 -> ReLU -> FC2

        ``[Decoder]`` is

        .. code-block:: text

            FC1 -> FC2 -> Sigmoid

        and ``[Sampler]`` draws a sample from the distribution

        .. math::

            \mathcal{N}(\\text{t_mean}, \exp \left\{\\text{t_log_var}\\right\} I)

        using the reparameterization trick.

        Parameters
        ----------
        T : int
            The dimension of the variational parameter `t`. Default is 5.
        enc_conv1_pad : int
            The padding for the first convolutional layer of the encoder. Default is 0.
        enc_conv1_stride : int
            The stride for the first convolutional layer of the encoder. Default is 1.
        enc_conv1_out_ch : int
            The number of output channels for the first convolutional layer of
            the encoder. Default is 32.
        enc_conv1_kernel_shape : tuple
            The number of rows and columns in each filter of the first
            convolutional layer of the encoder. Default is (5, 5).
        enc_pool1_kernel_shape : tuple
            The number of rows and columns in the receptive field of the first
            max pool layer of the encoder. Default is (2, 3).
        enc_pool1_stride : int
            The stride for the first MaxPool layer of the encoder. Default is
            2.
        enc_conv2_pad : int
            The padding for the second convolutional layer of the encoder.
            Default is 0.
        enc_conv2_out_ch : int
            The number of output channels for the second convolutional layer of
            the encoder. Default is 64.
        enc_conv2_kernel_shape : tuple
            The number of rows and columns in each filter of the second
            convolutional layer of the encoder. Default is (5, 5).
        enc_conv2_stride : int
            The stride for the second convolutional layer of the encoder.
            Default is 1.
        enc_pool2_stride : int
            The stride for the second MaxPool layer of the encoder. Default is
            1.
        enc_pool2_kernel_shape : tuple
            The number of rows and columns in the receptive field of the second
            max pool layer of the encoder. Default is (2, 3).
        latent_dim : int
            The dimension of the output for the first FC layer of the encoder.
            Default is 256.
        optimizer : str or :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object or None
            The optimization strategy to use when performing gradient updates.
            If None, use the :class:`~numpy_ml.neural_nets.optimizers.SGD`
            optimizer with default parameters. Default is "RMSProp(lr=0.0001)".
        init : str
            The weight initialization strategy. Valid entries are
            {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform',
            'std_normal', 'trunc_normal'}. Default is 'glorot_uniform'.
        """
        self.T = T
        self.init = init
        self.loss = VAELoss()
        self.optimizer = optimizer
        self.latent_dim = latent_dim
        self.enc_conv1_pad = enc_conv1_pad
        self.enc_conv2_pad = enc_conv2_pad
        self.enc_conv1_stride = enc_conv1_stride
        self.enc_conv1_out_ch = enc_conv1_out_ch
        self.enc_pool1_stride = enc_pool1_stride
        self.enc_conv2_out_ch = enc_conv2_out_ch
        self.enc_conv2_stride = enc_conv2_stride
        self.enc_pool2_stride = enc_pool2_stride
        self.enc_conv2_kernel_shape = enc_conv2_kernel_shape
        self.enc_pool2_kernel_shape = enc_pool2_kernel_shape
        self.enc_conv1_kernel_shape = enc_conv1_kernel_shape
        self.enc_pool1_kernel_shape = enc_pool1_kernel_shape

        self._init_params()

    def _init_params(self):
        self._dv = {}
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        """
        CNN encoder

        Conv1 -> ReLU -> MaxPool1 -> Conv2 -> ReLU -> MaxPool2 ->
            Flatten -> FC1 -> ReLU -> FC2
        """
        self.encoder = OrderedDict()
        self.encoder["Conv1"] = Conv2D(
            act_fn=ReLU(),
            init=self.init,
            pad=self.enc_conv1_pad,
            optimizer=self.optimizer,
            out_ch=self.enc_conv1_out_ch,
            stride=self.enc_conv1_stride,
            kernel_shape=self.enc_conv1_kernel_shape,
        )
        self.encoder["Pool1"] = Pool2D(
            mode="max",
            optimizer=self.optimizer,
            stride=self.enc_pool1_stride,
            kernel_shape=self.enc_pool1_kernel_shape,
        )
        self.encoder["Conv2"] = Conv2D(
            act_fn=ReLU(),
            init=self.init,
            pad=self.enc_conv2_pad,
            optimizer=self.optimizer,
            out_ch=self.enc_conv2_out_ch,
            stride=self.enc_conv2_stride,
            kernel_shape=self.enc_conv2_kernel_shape,
        )
        self.encoder["Pool2"] = Pool2D(
            mode="max",
            optimizer=self.optimizer,
            stride=self.enc_pool2_stride,
            kernel_shape=self.enc_pool2_kernel_shape,
        )
        self.encoder["Flatten3"] = Flatten(optimizer=self.optimizer)
        self.encoder["FC4"] = FullyConnected(
            n_out=self.latent_dim, act_fn=ReLU(), optimizer=self.optimizer
        )
        self.encoder["FC5"] = FullyConnected(
            n_out=self.T * 2,
            optimizer=self.optimizer,
            act_fn=Affine(slope=1, intercept=0),
            init=self.init,
        )

    def _build_decoder(self):
        """
        MLP decoder

        FC1 -> ReLU -> FC2 -> Sigmoid
        """
        self.decoder = OrderedDict()
        self.decoder["FC1"] = FullyConnected(
            act_fn=ReLU(),
            init=self.init,
            n_out=self.latent_dim,
            optimizer=self.optimizer,
        )
        # NB. `n_out` is dependent on the dimensionality of X. we use a
        # placeholder for now, and update it within the `forward` method
        self.decoder["FC2"] = FullyConnected(
            n_out=None, act_fn=Sigmoid(), optimizer=self.optimizer, init=self.init
        )

    @property
    def parameters(self):
        return {
            "components": {
                "encoder": {k: v.parameters for k, v in self.encoder.items()},
                "decoder": {k: v.parameters for k, v in self.decoder.items()},
            }
        }

    @property
    def hyperparameters(self):
        return {
            "layer": "BernoulliVAE",
            "T": self.T,
            "init": self.init,
            "loss": str(self.loss),
            "optimizer": self.optimizer,
            "latent_dim": self.latent_dim,
            "enc_conv1_pad": self.enc_conv1_pad,
            "enc_conv2_pad": self.enc_conv2_pad,
            "enc_conv1_in_ch": self.enc_conv1_in_ch,
            "enc_conv1_stride": self.enc_conv1_stride,
            "enc_conv1_out_ch": self.enc_conv1_out_ch,
            "enc_pool1_stride": self.enc_pool1_stride,
            "enc_conv2_out_ch": self.enc_conv2_out_ch,
            "enc_conv2_stride": self.enc_conv2_stride,
            "enc_pool2_stride": self.enc_pool2_stride,
            "enc_conv2_kernel_shape": self.enc_conv2_kernel_shape,
            "enc_pool2_kernel_shape": self.enc_pool2_kernel_shape,
            "enc_conv1_kernel_shape": self.enc_conv1_kernel_shape,
            "enc_pool1_kernel_shape": self.enc_pool1_kernel_shape,
            "encoder_ids": list(self.encoder.keys()),
            "decoder_ids": list(self.decoder.keys()),
            "components": {
                "encoder": {k: v.hyperparameters for k, v in self.encoder.items()},
                "decoder": {k: v.hyperparameters for k, v in self.decoder.items()},
            },
        }

    @property
    def derived_variables(self):
        dv = {
            "noise": None,
            "t_mean": None,
            "t_log_var": None,
            "dDecoder_FC1_in": None,
            "dDecoder_t_mean": None,
            "dEncoder_FC5_out": None,
            "dDecoder_FC1_out": None,
            "dEncoder_FC4_out": None,
            "dEncoder_Pool2_out": None,
            "dEncoder_Conv2_out": None,
            "dEncoder_Pool1_out": None,
            "dEncoder_Conv1_out": None,
            "dDecoder_t_log_var": None,
            "dEncoder_Flatten3_out": None,
            "components": {
                "encoder": {k: v.derived_variables for k, v in self.encoder.items()},
                "decoder": {k: v.derived_variables for k, v in self.decoder.items()},
            },
        }
        dv.update(self._dv)
        return dv

    @property
    def gradients(self):
        return {
            "components": {
                "encoder": {k: v.gradients for k, v in self.encoder.items()},
                "decoder": {k: v.gradients for k, v in self.decoder.items()},
            }
        }

    def _sample(self, t_mean, t_log_var):
        """
        Returns a sample from the distribution

            q(t | x) = N(t_mean, diag(exp(t_log_var)))

        using the reparameterization trick.

        Parameters
        ----------
        t_mean : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, latent_dim)`
            Mean of the desired distribution.
        t_log_var : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, latent_dim)`
            Log variance vector of the desired distribution.

        Returns
        -------
        samples: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, latent_dim)`
        """
        noise = np.random.normal(loc=0.0, scale=1.0, size=t_mean.shape)
        samples = noise * np.exp(t_log_var) + t_mean
        # save sampled noise for backward pass
        self._dv["noise"] = noise
        return samples

    def forward(self, X_train):
        """VAE forward pass"""
        if self.decoder["FC2"].n_out is None:
            fc2 = self.decoder["FC2"]
            self.decoder["FC2"] = fc2.set_params({"n_out": self.N})

        # assume each image is represented as a flattened row vector,
        n_ex, in_rows, N, in_ch = X_train.shape

        # encode the training batch to estimate the mean and variance of the
        # variational distribution
        out = X_train
        for k, v in self.encoder.items():
            out = v.forward(out)

        # extract the mean and log variance of the variational distribution
        # q(t | x) from the encoder output
        t_mean = out[:, : self.T]
        t_log_var = out[:, self.T :]

        # sample t from q(t | x) using reparamterization trick
        t = self._sample(t_mean, t_log_var)

        # pass the sampled latent value, t, through the decoder
        # to generate the average reconstruction
        X_recon = t
        for k, v in self.decoder.items():
            X_recon = v.forward(X_recon)

        self._dv["t_mean"] = t_mean
        self._dv["t_log_var"] = t_log_var
        return X_recon

    def backward(self, X_train, X_recon):
        """VAE backward pass"""
        n_ex = X_train.shape[0]
        D, E = self.decoder, self.encoder
        noise = self.derived_variables["noise"]
        t_mean = self.derived_variables["t_mean"]
        t_log_var = self.derived_variables["t_log_var"]

        # compute gradients through the VAE loss
        dY_pred, dLogVar, dMean = self.loss.grad(
            X_train.reshape(n_ex, -1), X_recon, t_mean, t_log_var
        )

        # backprop through the decoder
        dDecoder_FC1_out = D["FC2"].backward(dY_pred)
        dDecoder_FC1_in = D["FC1"].backward(dDecoder_FC1_out)

        # backprop through the sampler
        dDecoder_t_log_var = dDecoder_FC1_in * (noise * np.exp(t_log_var))
        dDecoder_t_mean = dDecoder_FC1_in

        # backprop through the encoder
        dEncoder_FC5_out = np.hstack(
            [dDecoder_t_mean + dMean, dDecoder_t_log_var + dLogVar]
        )
        dEncoder_FC4_out = E["FC5"].backward(dEncoder_FC5_out)
        dEncoder_Flatten3_out = E["FC4"].backward(dEncoder_FC4_out)
        dEncoder_Pool2_out = E["Flatten3"].backward(dEncoder_Flatten3_out)
        dEncoder_Conv2_out = E["Pool2"].backward(dEncoder_Pool2_out)
        dEncoder_Pool1_out = E["Conv2"].backward(dEncoder_Conv2_out)
        dEncoder_Conv1_out = E["Pool1"].backward(dEncoder_Pool1_out)
        dX = E["Conv1"].backward(dEncoder_Conv1_out)

        self._dv["dDecoder_t_mean"] = dDecoder_t_mean
        self._dv["dDecoder_FC1_in"] = dDecoder_FC1_in
        self._dv["dDecoder_FC1_out"] = dDecoder_FC1_out
        self._dv["dEncoder_FC5_out"] = dEncoder_FC5_out
        self._dv["dEncoder_FC4_out"] = dEncoder_FC4_out
        self._dv["dDecoder_t_log_var"] = dDecoder_t_log_var
        self._dv["dEncoder_Pool2_out"] = dEncoder_Pool2_out
        self._dv["dEncoder_Conv2_out"] = dEncoder_Conv2_out
        self._dv["dEncoder_Pool1_out"] = dEncoder_Pool1_out
        self._dv["dEncoder_Conv1_out"] = dEncoder_Conv1_out
        self._dv["dEncoder_Flatten3_out"] = dEncoder_Flatten3_out
        return dX

    def update(self, cur_loss=None):
        """Perform gradient updates"""
        for k, v in reversed(list(self.decoder.items())):
            v.update(cur_loss)
        for k, v in reversed(list(self.encoder.items())):
            v.update(cur_loss)
        self.flush_gradients()

    def flush_gradients(self):
        """Reset parameter gradients after update"""
        for k, v in self.decoder.items():
            v.flush_gradients()
        for k, v in self.encoder.items():
            v.flush_gradients()

    def fit(self, X_train, n_epochs=20, batchsize=128, verbose=True):
        """
        Fit the VAE to a training dataset.

        Parameters
        ----------
        X_train : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The input volume
        n_epochs : int
            The maximum number of training epochs to run. Default is 20.
        batchsize : int
            The desired number of examples in each training batch. Default is 128.
        verbose : bool
            Print batch information during training. Default is True.
        """
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batchsize = batchsize

        _, self.in_rows, self.in_cols, self.in_ch = X_train.shape
        self.N = self.in_rows * self.in_cols * self.in_ch

        prev_loss = np.inf
        for i in range(n_epochs):
            loss, estart = 0.0, time()
            batch_generator, nb = minibatch(X_train, batchsize, shuffle=True)

            # TODO: parallelize inner loop
            for j, b_ix in enumerate(batch_generator):
                bsize, bstart = len(b_ix), time()

                X_batch = X_train[b_ix]
                X_batch_col = X_train[b_ix].reshape(bsize, -1)

                X_recon = self.forward(X_batch)
                t_mean = self.derived_variables["t_mean"]
                t_log_var = self.derived_variables["t_log_var"]

                self.backward(X_batch, X_recon)
                batch_loss = self.loss(X_batch_col, X_recon, t_mean, t_log_var)
                loss += batch_loss

                self.update(batch_loss)

                if self.verbose:
                    fstr = "\t[Batch {}/{}] Train loss: {:.3f} ({:.1f}s/batch)"
                    print(fstr.format(j + 1, nb, batch_loss, time() - bstart))

            loss /= nb
            fstr = "[Epoch {}] Avg. loss: {:.3f}  Delta: {:.3f} ({:.2f}m/epoch)"
            print(fstr.format(i + 1, loss, prev_loss - loss, (time() - estart) / 60.0))
            prev_loss = loss
