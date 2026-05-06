import torch
from torch import nn
from pathlib import Path

class ResidualMLP(nn.Module):
    """

    A residual MLP model with skip connections and learnable scaling.

    f(x) = x + alpha * MLP(x)

    Parameters
    ----------
    d_in : int
        Input and output dimensionality.
    depth : int
        Number of hidden layers.
    d_hidden : int
        Dimensionality of hidden layers.
    dropout : float
        Dropout rate between layers.
    initial_alpha : float
        Initial value for the scaling parameter alpha.
    alpha_trainable : bool
        Whether alpha is trainable.
    dtype : torch.dtype
        Data type for the model parameters.
    """

    def __init__(
        self,
        d_in: int,
        depth: int = 2,
        d_hidden: int = 512,
        dropout: float = 0.1,
        initial_alpha: float = 0.1,
        alpha_trainable: bool = True,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()

        layers = []
        # layers.append(nn.LayerNorm(d_in, dtype=dtype))
        layers.append(nn.Linear(d_in, d_hidden, dtype=dtype))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        for _ in range(depth - 1):
            layers.append(nn.Linear(d_hidden, d_hidden, dtype=dtype))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(d_hidden, d_in, dtype=dtype))
        self.net = nn.Sequential(*layers)

        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=dtype))
        if not alpha_trainable:
            self.alpha.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with orthogonal initialization and biases to zero."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last = self.net[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return x + self.alpha * self.net(x)
    
    def forward_residual(self, x):
        return self.net(x)




def load_mlp_checkpoint(
    checkpoint_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.float,
):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = ResidualMLP(
        dtype=dtype,
        **ckpt["model_parameters"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt

