from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

import matplotlib.pyplot as plt

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 50000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
BUFFER_SIZE = int(1e6)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim))

        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.target = copy.deepcopy(self.model)
        for param in self.target.parameters():
            param.requires_grad = False

        self.memory = deque(maxlen=BUFFER_SIZE)
        random.seed(32)
        self.losses = []

    def calc_loss(self, batch):
        state, action, next_state, reward, done = batch
        q = self.model(state).gather(1, action)

        q_target = self.target(next_state).detach().max(1)[0].unsqueeze(1)

        loss = F.mse_loss(q, reward + GAMMA *
                          q_target*(1-done))
        self.losses.append(loss.data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.memory.appendleft(transition)

    def sample_batch(self):
        # Hints:
        # Sample batch from a replay buffer.
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster

        batch = random.sample(self.memory, BATCH_SIZE)

        states = torch.from_numpy(np.vstack([x[0] for x in batch])).float()
        actions = torch.from_numpy(np.vstack([x[1] for x in batch])).long()
        next_states = torch.from_numpy(
            np.vstack([x[2] for x in batch])).float()
        rewards = torch.from_numpy(np.vstack([x[3] for x in batch])).float()
        dones = torch.from_numpy(
            np.vstack([x[4] for x in batch]).astype(np.uint8)).float()

        return (states, actions, next_states, rewards, dones)

    # def soft_update(self):
    #     tau = 1e-4
    #     for target_param, local_param in zip(self.target.parameters(), self.model.parameters()):
    #         target_param.data.copy_(
    #             tau*local_param.data + (1.0-tau)*target_param.data)
    #         target_param.requires_grad = False

    def train_step(self, batch):
        # Use batch to update DQN's network.
        self.calc_loss(batch)

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        # pass
        self.target = copy.deepcopy(self.model)
        for param in self.target.parameters():
            param.requires_grad = False

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.

        state = torch.from_numpy(np.array(state))
        output = None
        self.model.eval()
        with torch.no_grad():
            if target:
                output = self.target.forward(state)
            else:
                output = self.model.forward(state)

        self.model.train()
        action = torch.argmax(output)
        return np.array(action.detach().numpy())

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.2  # !!!
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(
                f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()

    plt.plot(dqn.losses)
    plt.show()
