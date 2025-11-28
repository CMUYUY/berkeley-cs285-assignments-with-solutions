from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a
        # vectorized way. Make sure to 1) flatten obs, actions, rewards, terminals, and q_values, and 2) add
        # them to the info dict.
        # Flatten the lists of arrays into single arrays for vectorized operations
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        # Update the policy network using the calculated advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
        if self.critic is not None:
            # Perform multiple gradient steps on the critic
            critic_info: dict = {}
            for _ in range(self.baseline_gradient_steps):
                critic_update = self.critic.update(obs, q_values)
                for key, value in critic_update.items():
                    if key not in critic_info:
                        critic_info[key] = []
                    critic_info[key].append(value)
            
            # Average the critic losses across gradient steps
            for key in critic_info:
                critic_info[key] = np.mean(critic_info[key])
            
            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT: use the helper function self._discounted_return
        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        # HINT: use the helper function self._discounted_reward_to_go
        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        # Estimate the advantage when no value baseline is used.
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            # If no baseline, advantages are just the Q-values
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            # Use the critic to compute baseline values
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                # Advantages = Q-values - baseline values
                advantages = q_values - values
            else:
                # TODO: implement GAE-Lambda advantage calculation
                # HINT: append a dummy T+1 value for simpler recursive calculation
                # GAE-λ: A_t = sum_{l=0}^inf (gamma*lambda)^l * delta_{t+l}
                # where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
                batch_size = obs.shape[0]

                # Append dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    # δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
                    # A_t = δ_t + γ*λ*A_{t+1}
                    # Handle terminal states: if terminal, V(s_{t+1}) = 0
                    if terminals[i]:
                        delta = rewards[i] - values[i]
                        advantages[i] = delta
                    else:
                        delta = rewards[i] + self.gamma * values[i + 1] - values[i]
                        advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1]

                # Remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        # Normalize advantages to have mean 0 and std 1
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        # TODO: create a list of discounted returns of length len(rewards), where each entry t contains sum_{t'=0}^T gamma^t' r_{t'}
        # HINT: it is possible to write a vectorized solution, but a solution using a for loop is also fine
        # Compute total discounted return: G = Σ_{t=0}^{T-1} γ^t * r_t
        total_return = 0
        for t, reward in enumerate(rewards):
            total_return += (self.gamma ** t) * reward
        
        # Return a list where each element is the same total discounted return
        return [total_return] * len(rewards)


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        # TODO: create list of discounted rewards of length len(rewards), where the entry in each
        # index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        # Compute reward-to-go for each timestep using backward iteration
        # G_t = r_t + γ * G_{t+1}
        n = len(rewards)
        reward_to_go = [0] * n
        
        # Start from the end of the trajectory
        reward_to_go[-1] = rewards[-1]
        
        # Work backwards through the trajectory
        for t in reversed(range(n - 1)):
            reward_to_go[t] = rewards[t] + self.gamma * reward_to_go[t + 1]
        
        return reward_to_go
