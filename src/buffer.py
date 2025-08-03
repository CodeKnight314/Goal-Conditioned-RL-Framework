import numpy as np
from collections import deque
import torch
import random
from src.utils import RunningNormalizer


class ReplayBuffer:
    def __init__(self, max_len: int):
        self.buffer = deque(maxlen=max_len)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        assert len(self.buffer) >= batch_size, "Not enough in buffer to sample"

        batches = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batches)

        states = torch.stack([s.clone().detach() for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        next_states = torch.stack([s.clone().detach() for s in next_states]).to(
            self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PERBuffer:
    def __init__(self, max_len: int, alpha: float):
        self.buffer = deque(maxlen=max_len)
        self.priorities = deque(maxlen=max_len)
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epsilon = 1e-6

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(1.0)

    def sample(self, batch_size: int, beta: float):
        assert len(self) >= batch_size, "Not enough in buffer to sample"

        N = len(self)
        P = np.array(self.priorities, dtype=np.float32)
        P_sum = P.sum()
        if P_sum > 0:
            P /= P_sum
        else:
            P[:] = 1.0 / N

        indices = np.random.choice(N, batch_size, p=P)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        weights = (N * P[indices]) ** (-beta)
        weights /= weights.max()

        states = torch.stack([s.clone().detach() for s in states]).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        next_states = torch.stack([s.clone().detach() for s in next_states]).to(
            self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)
        weights = (
            torch.as_tensor(weights, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )

        return states, actions, rewards, next_states, dones, weights, indices

    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, indices, priorities):
        priorities = priorities.squeeze(-1)
        for index, priority in zip(indices, priorities):
            self.priorities[index] = (abs(priority) + self.epsilon) ** self.alpha


class HERBuffer:
    def __init__(
        self,
        max_mem_len: int,
        max_eps_len: int,
        nenvs: int,
        threshold: float = 0.05,
        k_future: int = 4,
    ):
        self.buffer = deque(maxlen=max_mem_len)
        self.episodes = [deque(maxlen=max_eps_len) for _ in range(nenvs)]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.k_future = k_future
        self.compute_reward = None
        self.obs_normalizer: RunningNormalizer = None
        self.dg_normalizer: RunningNormalizer = None

    def push(
        self, idx, state, action, next_state, reward, done, desired_goal, achieved_goal
    ):
        self.episodes[idx].append(
            (state, action, next_state, reward, done, desired_goal, achieved_goal)
        )

        if done or len(self.episodes[idx]) >= 50:
            self.apply_her(idx)
            self.episodes[idx].clear()

    def sample(self, batch_size: int):
        assert len(self.buffer) >= batch_size, "[ERROR] Not enough in buffer to sample"

        batches = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones, _, _ = zip(*batches)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def compute_termination(self, dg: np.array, ag: np.array):
        return np.linalg.norm(dg - ag, axis=-1) < self.threshold

    def apply_her(self, idx: int):
        eps_len = len(self.episodes[idx])

        for i, (s, a, ns, r, d, dg, ag) in enumerate(self.episodes[idx]):
            s = s.detach().cpu().numpy()
            ns = ns.detach().cpu().numpy()
            self.buffer.append((s, a, ns, r, d, dg, ag))

            for _ in range(self.k_future):
                if i < eps_len - 1:
                    future_idx = random.randint(i + 1, eps_len - 1)
                    _, _, _, _, _, _, future_ag = self.episodes[idx][future_idx]

                    new_desired_goal = np.array(future_ag, dtype=np.float32)
                    goal_dim = new_desired_goal.shape[0]

                    s_relabeled = np.concatenate(
                        [s[:-goal_dim], new_desired_goal], axis=-1
                    )
                    ns_relabeled = np.concatenate(
                        [ns[:-goal_dim], new_desired_goal], axis=-1
                    )

                    new_reward = self.compute_reward(ag, future_ag, {})
                    new_done = False

                    self.buffer.append(
                        (
                            s_relabeled,
                            a,
                            ns_relabeled,
                            new_reward,
                            new_done,
                            new_desired_goal,
                            ag,
                        )
                    )
