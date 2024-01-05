from torch import nn


def create_ff_network(layer_dims, h_activation='tanh', out_activation='none'):
    layers = []

    if h_activation == 'sigmoid':
        h_activation_fxn = nn.Sigmoid
    elif h_activation == 'relu':
        h_activation_fxn = nn.ReLU
    elif h_activation == 'none':
        h_activation_fxn = None
    else:
        h_activation_fxn = nn.Tanh

    for h in range(len(layer_dims) - 2):
        if h_activation_fxn is not None:
            layers.append(
                nn.Sequential(
                    nn.Linear(layer_dims[h], layer_dims[h + 1]),
                    h_activation_fxn()
                ))
        else:
            layers.append(nn.Linear(layer_dims[h], layer_dims[h + 1]))

    if out_activation == 'tanh':
        layers.append(
            nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]), nn.Tanh())
        )
    elif out_activation == 'sigmoid':
        layers.append(
            nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]), nn.Sigmoid())
        )
    elif out_activation == 'softmax':
        layers.append(
            nn.Sequential(nn.Linear(layer_dims[-2], layer_dims[-1]), nn.Softmax(dim=-1))
        )
    else:
        layers.append(
            nn.Linear(layer_dims[-2], layer_dims[-1])
        )

    return nn.Sequential(*layers)
