import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        # Convert observation to tensor and add batch dimension if needed
        if len(obs.shape) == 1:
            obs = obs[None]  # Add batch dimension
        
        obs_tensor = ptu.from_numpy(obs)
        action_distribution = self.forward(obs_tensor)
        
        # Sample action from the distribution
        action = action_distribution.sample()
        
        # Convert back to numpy
        action = ptu.to_numpy(action)
        
        # TODO: for discrete actions, return scalar (remove all dimensions)
        # For continuous actions, keep array shape but remove batch dimension
        if self.discrete:
            # Discrete: return scalar integer
            return int(action.item() if action.shape == () else action[0])
        else:
            # Continuous: remove batch dimension if we added it
            if len(action.shape) > 1 and action.shape[0] == 1:
                action = action[0]
            return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        # TODO: define the forward pass for a policy with a discrete action space.
        # TODO: define the forward pass for a policy with a continuous action space.
        if self.discrete:
            # For discrete action spaces, compute logits and return Categorical distribution
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            # For continuous action spaces, compute mean and use learned logstd
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)
        

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        # Get action distribution from current policy
        action_distribution = self.forward(obs)
        
        # Compute log probabilities of the taken actions
        log_probs = action_distribution.log_prob(actions)
        
        # Policy gradient loss: -E[log π(a|s) * A(s,a)]
        # We want to maximize log_prob weighted by advantage, so minimize negative
        loss = -(log_probs * advantages).mean()
        
        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
