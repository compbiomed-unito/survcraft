r"""Feature-to-parameter PyTorch modules for survival models.

Notes
-----
Inputs are floating feature tensors of shape ``(n_samples, input_size)``;
outputs are raw distribution parameters of shape ``(n_samples, output_size)``.
The caller manages device and dtype. Parameter constraints are applied by
survival modules, not these feature maps.
"""

import numpy
import torch


class FeedForwardNet(torch.nn.Module):
    r"""Map features to raw parameters through dense hidden layers.

    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of raw distribution parameters.
    hidden_sizes : sequence of int, default=[]
        Hidden layer widths; the default has no hidden layers.
    hidden_activation : callable, default=torch.nn.ReLU
        Zero-argument constructor called for each hidden activation.
    output_activation : callable, optional
        Zero-argument constructor for an optional output activation.
    batch_norm : bool, default=False
        Add batch normalization before each hidden activation.
    dropout : float, default=0.0
        Dropout probability after each hidden activation.

    Attributes
    ----------
    layers : torch.nn.Sequential
        Registered linear, normalization, activation, and dropout layers.

    Notes
    -----
    Each hidden block is linear, optional batch normalization, activation,
    then optional dropout. The final linear layer has only the optional
    output activation. With no hidden layers, this is a single linear map.
    Device and tensor contracts are described in the input_modules submodule.
    """
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=[],
        hidden_activation=torch.nn.ReLU,
        output_activation=None,
        batch_norm=False,
        dropout=0,
        # precision=torch.float32,
    ):
        r"""Initialize the feature map and its registered state.

        See Also
        --------
        FeedForwardNet : Constructor parameters.
        """
        super().__init__()
        layers = []
        for n, io_sizes in enumerate(
            zip([input_size, *hidden_sizes], [*hidden_sizes, output_size])
        ):
            layers.append(torch.nn.Linear(*io_sizes))
            if n < len(hidden_sizes):  # hidden layer
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(io_sizes[1]))
                layers.append(hidden_activation())
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
            else:  # output layer
                if output_activation is not None:
                    layers.append(output_activation())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X):
        r"""Map a feature batch to unconstrained distribution parameters.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, input_size)
            Floating features on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, output_size)
            Raw parameter values, before distribution transformations.
        """
        return self.layers(X)


class LinearFunctionInputModule(torch.nn.Module):
    r"""Apply a fixed affine map to feature tensors.

    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of raw distribution parameters.
    mode : {'identity', 'random'}, default='identity'
        Use a rectangular identity matrix or uniform random weights.
    multiplier : float, default=1.0
        Multiply the weight matrix by this value.
    shift : float, default=0.0
        Add this scalar to every output.
    use_first_n_feats : int, optional
        Set weight rows after this many features to zero.
    seed : int, optional
        Seed the local PyTorch generator in random mode.

    Attributes
    ----------
    linear_params : torch.Tensor, shape (input_size, output_size)
        Registered weight buffer, multiplied by multiplier.
    shift : float
        Scalar added to the output.

    Notes
    -----
    Computes :math:`Y=XW+s`. Identity mode uses a rectangular identity
    matrix; random mode uses a local generator and uniform [0, 1) weights.
    Rows after use_first_n_feats are zeroed before applying multiplier.
    Weights are buffers and are not optimized. Unknown modes are not
    validated explicitly and currently fail during construction.
    """
    def __init__(
        self,
        input_size,
        output_size,
        mode="identity",
        multiplier=1.0,
        shift=0.0,
        use_first_n_feats=None,
        seed=None,
    ):
        r"""Initialize the feature map and its registered state.

        See Also
        --------
        LinearFunctionInputModule : Constructor parameters.
        """
        super().__init__()

        if mode == "identity":
            lin_params = torch.eye(input_size, output_size)
        elif mode == "random":
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(seed)
            lin_params = torch.rand(input_size, output_size, generator=g)

        if use_first_n_feats is not None:
            lin_params[use_first_n_feats:] = 0.0

        self.register_buffer("linear_params", multiplier * lin_params)
        self.shift = shift

    def forward(self, x):
        r"""Map a feature batch to unconstrained distribution parameters.

        Parameters
        ----------
        x : torch.Tensor, shape (n_samples, input_size)
            Floating features on the module device.

        Returns
        -------
        torch.Tensor, shape (n_samples, output_size)
            Raw parameter values, before distribution transformations.
        """
        return torch.matmul(x, self.linear_params) + self.shift
