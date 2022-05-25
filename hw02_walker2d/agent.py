import random

import numpy as np
import torch

try:
    from train import Actor
except ImportError:
    from .train import Actor

random.seed(1312)
torch.manual_seed(1312)
np.random.seed(1312)


class Agent:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(
            __file__[:-8] + "/agent.pkl", map_location=self.device)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float().to(self.device)
            action, _, _ = self.model.act(state)
        return action.cpu().numpy()

    def reset(self):
        pass

# a = Agent()
# print(a.act(list([0]*22)))
