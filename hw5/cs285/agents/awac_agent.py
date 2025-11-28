from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            # Get actor's action distribution for next states
            next_action_dist = self.actor(next_observations)
            
            # Get Q-values for all actions in next states
            next_qa_values = self.target_critic(next_observations)

            # Use the actor to compute a critic backup
            # Compute E[Q(s', a)] = sum_a pi(a|s') * Q(s', a)
            next_action_probs = next_action_dist.probs
            next_qs = (next_action_probs * next_qa_values).sum(dim=1)

            # TODO(student): Compute the TD target
            # Convert dones to float to avoid PyTorch bool subtraction error
            target_values = rewards + self.discount * next_qs * (1 - dones.float())

        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        with torch.no_grad():
            # Get Q-values for all actions
            qa_values = self.critic(observations)
            
            # Get Q-value for the specific action taken
            q_values = qa_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Get actor's action distribution
            if action_dist is None:
                action_dist = self.actor(observations)
            
            # Compute expected Q-value: E[Q(s, a)] = sum_a pi(a|s) * Q(s, a)
            action_probs = action_dist.probs
            values = (action_probs * qa_values).sum(dim=1)

            # Advantage = Q(s, a) - E[Q(s, a)]
            advantages = q_values - values
        
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        # Compute advantages
        action_dist = self.actor(observations)
        advantages = self.compute_advantage(observations, actions, action_dist)
        
        # Compute AWAC weights: exp(A(s, a) / temperature)
        weights = torch.exp(advantages / self.temperature)
        
        # Compute weighted log probability loss
        log_probs = action_dist.log_prob(actions)
        loss = -(weights * log_probs).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
