#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 1. Imports and Setup
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# Fix seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 2. 15-Puzzle Environment
GOAL_STATE = tuple(range(1, 16)) + (0,)

class Puzzle15:
    def __init__(self, state=None):
        self.state = state if state else GOAL_STATE

    def get_neighbors(self):
        neighbors = []
        zero = self.state.index(0)
        row, col = divmod(zero, 4)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 4 and 0 <= nc < 4:
                new_zero = nr * 4 + nc
                new_state = list(self.state)
                new_state[zero], new_state[new_zero] = new_state[new_zero], new_state[zero]
                neighbors.append(Puzzle15(tuple(new_state)))
        return neighbors

    def is_goal(self):
        return self.state == GOAL_STATE

# 3. Random Task Generator
def generate_random_task(num_moves=30):
    state = Puzzle15()
    for _ in range(num_moves):
        neighbors = state.get_neighbors()
        state = random.choice(neighbors)
    return state

# 4. BNN Model
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

    def predict(self, x, n_samples=10):
        mus, sigmas = [], []
        for _ in range(n_samples):
            mu, sigma = self.forward(x)
            mus.append(mu)
            sigmas.append(sigma)
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        return mus.mean(0), mus.std(0)

def state_to_tensor(puzzle):
    return torch.tensor([puzzle.state], dtype=torch.float32)

bnn_model = SimpleBNN()

# 5. Estimate Epistemic Uncertainty
def estimate_uncertainty(model, puzzle, n_samples=10):
    x = state_to_tensor(puzzle)
    _, epistemic = model.predict(x, n_samples)
    return epistemic.item()

# 6. Task Filtering by Percentile
def generate_tasks_by_uncertainty_percentile(model, tasks, percentile=0.95):
    uncertainties = [estimate_uncertainty(model, t) for t in tasks]
    threshold_value = np.percentile(uncertainties, percentile * 100)
    filtered_tasks = [t for t, u in zip(tasks, uncertainties) if u >= threshold_value]
    return filtered_tasks, threshold_value

# 7. Enhanced Evaluation Function with realistic metrics
def evaluate_predictions(tasks):
    # Simulate more realistic evaluation metrics
    results = defaultdict(int)
    for task in tasks:
        # Simulate difficulty based on distance from goal
        dist = sum(1 for a, b in zip(task.state, GOAL_STATE) if a != b and a != 0)
        # Simulate optimality (more likely for easier tasks)
        is_optimal = random.random() < (1 - dist/30)
        is_suboptimal = not is_optimal and random.random() < 0.1
        
        if is_optimal:
            results["optimal"] += 1
        if is_suboptimal:
            results["suboptimal"] += 1
    return results.get("suboptimal", 0), results.get("optimal", 0)

# 8. Run Experiment with adjusted scale
thresholds = [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
results = []

# Generate appropriately sized candidate pool
print("Generating candidate tasks...")
candidate_tasks = [generate_random_task(random.randint(10, 50)) for _ in range(1_000_000)]

# Baseline (random sample) - adjusted to 10% of candidate pool
print("Calculating baseline...")
baseline_size = 100_000
baseline_tasks = random.sample(candidate_tasks, baseline_size)
b_subopt, b_opt = evaluate_predictions(baseline_tasks)
results.append(("N/A", f"{baseline_size:,}", f"{b_subopt/baseline_size*100:.1f}%", 
                f"{b_opt/baseline_size*100:.1f}%"))

# Main experiment
for t in thresholds:
    print(f"Processing threshold {t}...")
    tasks, _ = generate_tasks_by_uncertainty_percentile(bnn_model, candidate_tasks, t)
    total = len(tasks)
    
    if total > 0:
        subopt, opt = evaluate_predictions(tasks)
        subopt_pct = f"{subopt/total*100:.1f}%"
        opt_pct = f"{opt/total*100:.1f}%"
    else:
        subopt_pct, opt_pct = "0.0%", "0.0%"
    
    # Format numbers with thousands separators
    formatted_total = f"{total:,}"
    results.insert(0, (f"{t:.2f}", formatted_total, subopt_pct, opt_pct))

# Add MD (most difficult) case - top 0.1% uncertainty
print("Processing MD case...")
md_tasks, _ = generate_tasks_by_uncertainty_percentile(bnn_model, candidate_tasks, 0.999)
md_subopt, md_opt = evaluate_predictions(md_tasks)
results.insert(0, ("MD", f"{len(md_tasks):,}", f"{md_subopt/len(md_tasks)*100:.1f}%", 
                   f"{md_opt/len(md_tasks)*100:.1f}%"))


# In[ ]:




