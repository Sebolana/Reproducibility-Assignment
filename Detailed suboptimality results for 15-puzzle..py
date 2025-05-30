#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import copy

GOAL_STATE = tuple(range(16))

class Puzzle15:
    def __init__(self, state=None):
        if state is None:
            state = GOAL_STATE
        self.state = tuple(state)
    
    def is_goal(self):
        return self.state == GOAL_STATE

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

    def to_input(self):
        return np.array(self.state).reshape(4, 4) / 15.0


# In[2]:


def generate_random_task(num_moves=30):
    state = Puzzle15(GOAL_STATE)
    for _ in range(num_moves):
        neighbors = state.get_neighbors()
        state = random.choice(neighbors)
    return state


# In[3]:


class DummyBNN:
    def predict(self, x):
        return random.uniform(10, 40)  # Dummy heuristic

    def predict_epistemic_uncertainty(self, x):
        return random.uniform(0, 1)  # Dummy uncertainty

bnn_model = DummyBNN()


# In[4]:


def generate_tasks_by_uncertainty_percentile(model, candidate_tasks, percentile):
    inputs = [task.to_input() for task in candidate_tasks]
    uncertainties = [model.predict_epistemic_uncertainty(inp) for inp in inputs]

    threshold_value = np.percentile(uncertainties, percentile * 100)
    selected = [task for task, unc in zip(candidate_tasks, uncertainties) if unc >= threshold_value]

    return selected, threshold_value


# In[5]:


def evaluate_predictions(model, tasks):
    subopt = 0
    opt = 0
    for task in tasks:
        pred = model.predict(task.to_input())
        if pred <= 18:  # Optimal threshold
            opt += 1
        if pred > 18 and pred < 30:  # Suboptimal threshold
            subopt += 1
    return subopt, opt


# In[6]:


thresholds = [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
results = []

# Pre-generate a shared pool of random tasks
candidate_tasks = [generate_random_task() for _ in range(1000)]

for t in thresholds:
    tasks, _ = generate_tasks_by_uncertainty_percentile(bnn_model, candidate_tasks, t)
    total = len(tasks)
    if total > 0:
        subopt, opt = evaluate_predictions(bnn_model, tasks)
        results.append({
            "Thresh": t,
            "Generated": total,
            "Subopt": f"{(subopt / total) * 100:.1f}%",
            "Optimal": f"{(opt / total) * 100:.1f}%"
        })
    else:
        results.append({
            "Thresh": t,
            "Generated": 0,
            "Subopt": "N/A",
            "Optimal": "N/A"
        })

# Baseline (random sample)
baseline_tasks = random.sample(candidate_tasks, 100)
subopt, opt = evaluate_predictions(bnn_model, baseline_tasks)
results.append({
    "Thresh": "N/A",
    "Generated": 100,
    "Subopt": f"{(subopt / 100) * 100:.1f}%",
    "Optimal": f"{(opt / 100) * 100:.1f}%"
})


# In[7]:


import pandas as pd

df = pd.DataFrame(results)
print(df.to_string(index=False))


# In[ ]:




