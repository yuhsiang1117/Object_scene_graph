"""The discrete (habitat) action head for the vendored PointNav ResNet.

`nh_pointnav_policy.PointNavResNetPolicy` is VLFM's Spot variant: a Gaussian
head over (linear, angular) velocity. `pointnav_weights.pth` -- the checkpoint
ASCENT runs -- is the discrete habitat policy instead, so it needs a
`Categorical` head over the four navigation actions. See README.md for the key
evidence; the encoder body is shared verbatim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .nh_pointnav_policy import PointNavResNetNet

# habitat's ObjectNav action ids, and the order the policy was trained in.
ACTION_NAMES = ("stop", "move_forward", "turn_left", "turn_right")
NUM_ACTIONS = len(ACTION_NAMES)
HIDDEN_SIZE = 512


class CategoricalNet(nn.Module):
    """Named to match the checkpoint (`action_distribution.linear.*`)."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.linear(x).float())


class CriticHead(nn.Module):
    """Value head. Unused at inference, defined so the checkpoint loads
    strictly -- a silently-dropped key is how a wrong head goes unnoticed."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class PointNavResNetDiscretePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = PointNavResNetNet(discrete_actions=True, no_fwd_dict=True)
        self.action_distribution = CategoricalNet(HIDDEN_SIZE, NUM_ACTIONS)
        self.critic = CriticHead(HIDDEN_SIZE)

    @property
    def num_recurrent_layers(self) -> int:
        return self.net.num_recurrent_layers

    @torch.no_grad()
    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        if deterministic:
            action = distribution.probs.argmax(dim=-1, keepdim=True)
        else:
            action = distribution.sample().unsqueeze(-1)
        return action, rnn_hidden_states


def rename_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Checkpoint -> this module's key names.

    The checkpoint predates habitat-baselines splitting the previous-action
    embedding into discrete/continuous variants, so it carries the old flat
    name. VLFM applies the mirror-image rename for its continuous policy
    (`vlfm/policy/utils/pointnav_policy.py:181-190`).
    """
    return {
        k.replace("net.prev_action_embedding.", "net.prev_action_embedding_discrete."): v
        for k, v in state_dict.items()
    }


def load_pointnav_policy(path: str | Path) -> PointNavResNetDiscretePolicy:
    """Load the converted, weights-only checkpoint written by
    `scripts/download_weights.py --pointnav`.

    `weights_only=True` is deliberate: the raw VLFM file pickles a
    habitat-baselines config object, which neither loads here (the package is
    not installed) nor should be executed at eval time. The conversion strips it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PointNav weights not found at {path}. "
            "Run: python scripts/download_weights.py --pointnav"
        )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:  # a raw, unconverted VLFM checkpoint
        state_dict = state_dict["state_dict"]
    policy = PointNavResNetDiscretePolicy()
    policy.load_state_dict(rename_checkpoint_keys(state_dict))
    policy.eval()
    return policy
