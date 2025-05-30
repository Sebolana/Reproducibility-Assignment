import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

GOAL_STATE = tuple(range(1, 16)) + (0,)

class Puzzle15:
    def __init__(self, state=None):
        self.state = state if state else GOAL_STATE

    def get_neighbors(self):
        def swap(state, i, j):
            lst = list(state)
            lst[i], lst[j] = lst[j], lst[i]
            return tuple(lst)

        neighbors = []
        zero_index = self.state.index(0)
        x, y = divmod(zero_index, 4)
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4:
                ni = nx * 4 + ny
                neighbors.append(Puzzle15(swap(self.state, zero_index, ni)))
        return neighbors

class SimpleBNN(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, 1)
        self.fc2_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        sigma = F.softplus(self.fc2_sigma(h)) + 1e-6
        return mu, sigma

def generate_random_task(num_moves=30):
    state = Puzzle15()
    for _ in range(num_moves):
        neighbors = state.get_neighbors()
        state = random.choice(neighbors)
    return state

if __name__ == "__main__":
    bnn_model = SimpleBNN()
    candidate_tasks = [generate_random_task() for _ in range(1000)]
    print(f"Generated {len(candidate_tasks)} candidate puzzles")
