from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        # TODO(student): modify the loss to implement CQL
        # Hint: `variables` includes qa_values and q_values from your CQL implementation
        qa_values = variables["qa_values"]  # Q(s, a) for all actions
        
        # Compute CQL penalty: logsumexp(Q(s, :) / temperature) * temperature - Q(s, a_data)
        q_logsumexp = torch.logsumexp(qa_values / self.cql_temperature, dim=1)
        q_data = qa_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        cql_penalty = (q_logsumexp * self.cql_temperature - q_data).mean()
        
        # Add CQL penalty to loss
        loss = loss + self.cql_alpha * cql_penalty
        
        # Log CQL metrics
        metrics["cql_penalty"] = cql_penalty.item()
        metrics["q_logsumexp"] = q_logsumexp.mean().item()

        return loss, metrics, variables
