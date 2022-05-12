import random
import numpy as np
import os
import torch
from gym import make


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.eval()
        random.seed(32)

    def act(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        q = self.model(state).data.numpy()
        # print(q[0])
        return int(np.argmax(q))

# a = Agent()
# env = make("LunarLander-v2")
# state = env.reset()
# print(state)
# print(a.act(state))