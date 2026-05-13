import numpy
import torch


class FeedForwardNet(torch.nn.Module):
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
        """Create a feed-forward net

        sizes: sequence of integers
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
        return self.layers(X)


class LinearFunctionInputModule(torch.nn.Module):
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
        return torch.matmul(x, self.linear_params) + self.shift
