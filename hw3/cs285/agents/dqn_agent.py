from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Explore: choose random action
            action = torch.tensor(np.random.randint(self.num_actions))
        else:
            # Exploit: choose action with highest Q-value
            with torch.no_grad():
                qa_values = self.critic(observation)  # Shape: (1, num_actions)
                action = qa_values.argmax(dim=1)  # Get action with max Q-value

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            # Compute Q-values for all actions in next states
            next_qa_values = self.target_critic(next_obs)  # Shape: (batch_size, num_actions)

            if self.use_double_q:
                # Double DQN: use online network to select action, target network to evaluate
                next_qa_values_online = self.critic(next_obs)
                next_action = next_qa_values_online.argmax(dim=1)  # Select action using online network
                # Gather Q-values from target network for selected actions
                next_q_values = next_qa_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            else:
                # Vanilla DQN: select action with max Q-value from target network
                next_action = next_qa_values.argmax(dim=1)
                next_q_values = next_qa_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            
            # Compute target values: r + gamma * max_a' Q(s', a') * (1 - done)
            target_values = reward + self.discount * next_q_values * (1 - done.float())

        # TODO(student): train the critic with the target values
        # Get Q-values for all state-action pairs
        qa_values = self.critic(obs)  # Shape: (batch_size, num_actions)
        # Extract Q-values for the actions that were taken
        q_values = qa_values.gather(1, action.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)
        # Compute loss
        loss = self.critic_loss(q_values, target_values)


        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        # Update the critic network
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        
        # Update target network periodically
        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
