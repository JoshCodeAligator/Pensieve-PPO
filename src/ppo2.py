import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, Any

from config_layered import ENTROPY_W, LR


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)  # logits (un-normalized)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.val = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.val(x).squeeze(-1)  # [B]


class Network(nn.Module):
    """
    Minimal A2C-compatible network with the API expected by train.py/test.py.
    - act()         -> sample action with optional action mask
    - predict()     -> action probabilities for 1 state (optional mask)
    - value()       -> V(s)
    - train(...)    -> single A2C update step given (s, a(one-hot), old_p, v_target)
    - compute_v(...)-> bootstrap helper kept for legacy compatibility
    - get/set/save/load params helpers
    """
    def __init__(self, state_dim: int, action_dim: int, learning_rate: Optional[float] = None):
        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        lr = float(learning_rate) if learning_rate is not None else float(LR)
        self.optimizer = optim.Adam( # type: ignore
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.entropy_w = float(ENTROPY_W)

    # ---------- Inference ----------
    def _ensure_batch(self, s) -> torch.Tensor:
        if isinstance(s, torch.Tensor):
            t = s.float()
        else:
            t = torch.tensor(s, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def _apply_mask_to_logits(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        # mask is 1 for valid, 0 for invalid; add log(mask+eps) -> -inf for invalid
        return logits + (mask + 1e-8).log()

    def act(self, s, mask: Optional[torch.Tensor] = None):
        """
        Returns:
          a:      [B] sampled integer actions
          logpa:  [B] log-prob of sampled actions
          ent:    []  mean entropy
        """
        s_t = self._ensure_batch(s)
        logits = self.actor(s_t)
        logits = self._apply_mask_to_logits(logits, mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy().mean()

    def predict(self, s, mask: Optional[torch.Tensor] = None):
        """
        Returns action probabilities (numpy) for a single state (or batch).
        """
        s_t = self._ensure_batch(s)
        with torch.no_grad():
            logits = self.actor(s_t)
            logits = self._apply_mask_to_logits(logits, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs.squeeze(0) if probs.shape[0] == 1 else probs

    def value(self, s):
        s_t = self._ensure_batch(s)
        with torch.no_grad():
            v = self.critic(s_t).cpu().numpy()
        return v.squeeze(0) if v.shape[0] == 1 else v

    # ---------- Training / Updates ----------
    def compute_v(self, s_batch, a_batch, r_batch, done):
        """
        Bootstrap V(s_T) to align shapes with r_batch. Kept for legacy compatibility.
        """
        import numpy as np
        s_last = self._ensure_batch(s_batch[-1])
        with torch.no_grad():
            v_last = float(self.critic(s_last).item())
        return np.array([v_last for _ in r_batch], dtype=np.float32)

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch: int = 0) -> Dict[str, Any]:
        """
        A2C step:
          s_batch: [N, S_DIM]
          a_batch: [N, A_DIM] one-hot chosen actions
          p_batch: [N, A_DIM] (unused but kept for API compatibility)
          v_batch: [N]       target returns (discounted reward)
        """
        # tensors
        s = self._ensure_batch(s_batch)
        a = torch.tensor(a_batch, dtype=torch.float32)
        v_targ = torch.tensor(v_batch, dtype=torch.float32).view(-1)

        # forward
        logits = self.actor(s)                  # [N, A]
        V_s    = self.critic(s)                 # [N]
        logp   = F.log_softmax(logits, dim=-1)  # [N, A]
        probs  = torch.softmax(logits, dim=-1)  # [N, A]
        ent    = -(probs * logp).sum(dim=-1).mean()

        # log prob of chosen actions (via one-hot)
        logp_a = (logp * a).sum(dim=-1)         # [N]

        # advantage signal
        adv = (v_targ - V_s).detach()

        # losses
        policy_loss = -(logp_a * adv).mean()
        value_loss  = 0.5 * (v_targ - V_s).pow(2).mean()
        loss = policy_loss + value_loss - self.entropy_w * ent

        # step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(ent.item()),
        }

    # ---------- Params I/O ----------
    def get_network_params(self):
        return {
            "actor": {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
            "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
        }

    def set_network_params(self, params):
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])

    def save_model(self, path: str):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load_model(self, path: str, map_location: Optional[str] = "cpu"):
        sd = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])