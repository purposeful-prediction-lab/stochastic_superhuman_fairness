import torch
import torch.nn as nn
import torch.optim as optim
from stochastic_superhuman_fairness.core.models.base_model import BaseModel


class PPOModel(BaseModel):
    """
    PPO-style actor-critic for subdominance-driven RL phases.
    Currently placeholder structure compatible with Learner.
    """

    def __init__(self, cfg, demonstrator):
        super().__init__(cfg, demonstrator)
        input_dim = demonstrator.dataset["metadata"]["n_features"]
        hidden = cfg.get("hidden_size", 128)
        lr = cfg.get("lr", 3e-4)
        self.clip_range = cfg.get("clip_range", 0.2)
        self.value_coef = cfg.get("value_coef", 0.5)

        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        return self.policy(X)

    def train_step(self, X, y):
        """
        Placeholder PPO-style loss (acts like supervised BCE for now).
        Will later include advantage and clipped objective.
        """
        self.opt.zero_grad(set_to_none=True)
        logits = self.policy(X)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y)
        loss.backward()
        self.opt.step()
        return {"loss": float(loss.item())}
