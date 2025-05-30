# Reproducibility-Assignment
# 15-Puzzle Analysis: Reproducibility Study

![15-Puzzle Example](https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/15-puzzle.svg/200px-15-puzzle.svg.png)  
*Classic 15-puzzle (image from Wikipedia)*

This repository compares three approaches to solving the 15-puzzle problem, focusing on reproducibility in AI research.

Reproducibility-Assignment/
├── basic_analysis/ # Classical solver with Manhattan distance
│ └── puzzle15_basic.py # → Measures solvability and heuristic performance
├── bnn_analysis/ # Bayesian Neural Network implementation
│ └── puzzle15_bnn.py # → Evaluates epistemic uncertainty
├── dummy_bnn_analysis/ # Control group for comparison
│ └── puzzle15_dummy.py # → Provides baseline metrics
├── requirements.txt # Dependency specifications
└── README.md # This document
## Requirements
Python 3.7+ with:
```bash
pip install -r requirements.txt

