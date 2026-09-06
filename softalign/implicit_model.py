import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitMLP(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=1, num_layers=4):
        """
        A simple feedforward network with 'num_layers' layers.
        in_dim: number of input dimensions (x, y, t)
        hidden_dim: width of hidden layers
        out_dim: output dimension (scalar F)
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, coords):
        # coords: [N, 3] tensor with columns (x, y, t)
        return self.network(coords)

class EventFrameAlignmentModel(nn.Module):
    minimum_divisor = 1e-5

    def __init__(
        self,
        dt_init=0.1,
        hidden_dim=128,
        num_layers=4,
        scale_init=0.8,
    ):
        super().__init__()
        if dt_init <= self.minimum_divisor or scale_init <= self.minimum_divisor:
            raise ValueError(
                f"dt_init and scale_init must exceed {self.minimum_divisor}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.F_net = ImplicitMLP(
            in_dim=3,
            hidden_dim=self.hidden_dim,
            out_dim=1,
            num_layers=self.num_layers,
        )

        # Learnable threshold parameter for the event branch
        self.threshold = nn.Parameter(torch.tensor(0.0))

        # Positive divisors are represented through softplus so training cannot
        # cross zero and produce unstable coordinate transforms.
        self.raw_dt = nn.Parameter(self._inverse_softplus(dt_init - self.minimum_divisor))

        # Learnable affine transformation parameters for events:
        # Scale factor applied to x and y (spatial scaling)
        self.raw_scale = nn.Parameter(
            self._inverse_softplus(scale_init - self.minimum_divisor)
        )

        # Translations in x and y respectively
        self.shift_x = nn.Parameter(torch.tensor(0.05))  # Initialize to 0.05 as specified
        self.shift_y = nn.Parameter(torch.tensor(0.08))  # Initialize to 0.08 as specified

        # Translation in time (temporal shift)
        self.shift_t = nn.Parameter(torch.tensor(1.0))  # Initialize to 1s as specified

    def forward_F(self, coords):
        """Evaluate the implicit network F at given coordinates."""
        return self.F_net(coords)  # returns F(x,y,t)

    @staticmethod
    def _inverse_softplus(value):
        scalar = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(scalar))

    @property
    def scale(self):
        return F.softplus(self.raw_scale) + self.minimum_divisor

    @property
    def dt(self):
        return F.softplus(self.raw_dt) + self.minimum_divisor

    def forward_event(self, coords):
        """
        Given event data coordinates [N,3] with (x,y,t),
        compute the event response sigmoid(dF/dt - threshold)
        where dF/dt is approximated as [F(x,y,t+dt) - F(x,y,t)]/dt

        First apply the affine transformation to events:
            x' = x / scale + shift_x
            y' = y / scale + shift_y
            t' = t + shift_t
        """
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("event coordinates must have shape N x 3")
        x_transformed = coords[:, 0:1] / self.scale + self.shift_x
        y_transformed = coords[:, 1:2] / self.scale + self.shift_y
        t_transformed = coords[:, 2:3] + self.shift_t

        # Concatenate to create transformed coordinates
        coords_t = torch.cat([x_transformed, y_transformed, t_transformed], dim=1)

        # Create coordinates for t+dt
        t_plus_dt = t_transformed + self.dt
        coords_t_plus_dt = torch.cat([x_transformed, y_transformed, t_plus_dt], dim=1)

        # Evaluate F at t and t+dt
        F_t = self.F_net(coords_t)
        F_t_plus_dt = self.F_net(coords_t_plus_dt)

        # Approximate dF/dt and apply sigmoid activation with threshold
        dF_dt = (F_t_plus_dt - F_t) / self.dt
        event_response = torch.sigmoid(dF_dt - self.threshold)

        return event_response

    def forward_frame(self, coords):
        """
        For frame data coordinates [N,3] with (x,y,t),
        directly evaluate the implicit function F(x,y,t).
        Frame data are considered as the reference standard.
        """
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("frame coordinates must have shape N x 3")
        return self.F_net(coords)

    def config(self):
        return {
            "dt_init": float(self.dt.detach().cpu()),
            "scale_init": float(self.scale.detach().cpu()),
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }


# Compatibility alias retained for notebooks that imported the original name.
MLP_F = ImplicitMLP
