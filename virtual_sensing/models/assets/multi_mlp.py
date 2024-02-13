from tsl.nn.blocks import MLP
from tsl.nn.layers.multi import MultiDense, MultiLinear
import torch.nn as nn

class MultiMLP(MLP):
    """Multi-layer Perceptron encoder with optional linear readout and 
        different weight for each instance.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int, optional): Size of the optional readout.
        exog_size (int, optional): Size of the optional exogenous variables.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        activation (str, optional): Activation function. (default: `relu`)
        dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                input_size: int,
                hidden_size: int,
                n_instances: int,
                output_size: int = None,
                pattern: str = None,
                instance_dim: int = -2,
                channel_dim: int = -1,
                exog_size: int = None,
                n_layers: int = 1,
                activation: str = 'relu',
                dropout=0.):
        super(MultiMLP, self).__init__(input_size=input_size,
                                        hidden_size=hidden_size,
                                        output_size=output_size,
                                        exog_size=exog_size,
                                        n_layers=n_layers,
                                        activation=activation,
                                        dropout=dropout)
        if exog_size is not None:
            input_size += exog_size
        mlp_layers = [      
                        MultiDense(in_channels = input_size if i == 0 else hidden_size,
                                    out_channels = hidden_size,
                                    n_instances = n_instances,
                                    activation = activation,
                                    pattern = pattern, 
                                    instance_dim = instance_dim,
                                    channel_dim = channel_dim) for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*mlp_layers)
        if output_size is not None:
            self.readout = MultiLinear(in_channels = hidden_size,
                                        out_channels = output_size,
                                        n_instances = n_instances,
                                        pattern = pattern, 
                                        instance_dim = instance_dim,
                                        channel_dim = channel_dim)
        else:
            self.register_parameter('readout', None)

    def forward(self, x, u=None):
        return super().forward(x, u)

        