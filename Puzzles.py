#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import time
from typing import Tuple, List

class Puzzle15:
    def __init__(self, state=None):
        self.size = 4
        self.goal = np.arange(1, 17).reshape(4, 4) % 16
        self.state = self.goal.copy() if state is None else np.array(state)
        
    def shuffle(self, moves: int = 20):
        """Shuffle the puzzle with valid moves starting from solved state"""
        self.state = self.goal.copy()  # Reset to solved state
        for _ in range(moves):
            blank = self.find_blank()
            possible_moves = []
            if blank[0] > 0: possible_moves.append('UP')
            if blank[0] < 3: possible_moves.append('DOWN')
            if blank[1] > 0: possible_moves.append('LEFT')
            if blank[1] < 3: possible_moves.append('RIGHT')
            self.move(random.choice(possible_moves))
    
    def find_blank(self) -> Tuple[int, int]:
        return tuple(np.argwhere(self.state == 0)[0])
    
    def move(self, direction: str) -> bool:
        blank = self.find_blank()
        new_state = self.state.copy()
        
        if direction == 'UP' and blank[0] > 0:
            new_state[blank[0], blank[1]], new_state[blank[0]-1, blank[1]] =                 new_state[blank[0]-1, blank[1]], new_state[blank[0], blank[1]]
        elif direction == 'DOWN' and blank[0] < 3:
            new_state[blank[0], blank[1]], new_state[blank[0]+1, blank[1]] =                 new_state[blank[0]+1, blank[1]], new_state[blank[0], blank[1]]
        elif direction == 'LEFT' and blank[1] > 0:
            new_state[blank[0], blank[1]], new_state[blank[0], blank[1]-1] =                 new_state[blank[0], blank[1]-1], new_state[blank[0], blank[1]]
        elif direction == 'RIGHT' and blank[1] < 3:
            new_state[blank[0], blank[1]], new_state[blank[0], blank[1]+1] =                 new_state[blank[0], blank[1]+1], new_state[blank[0], blank[1]]
        else:
            return False
        
        self.state = new_state
        return True
    
    def is_solved(self) -> bool:
        return np.array_equal(self.state, self.goal)
    
    def manhattan_distance(self) -> int:
        distance = 0
        for i in range(4):
            for j in range(4):
                val = self.state[i,j]
                if val != 0:
                    target_i, target_j = (val-1) // 4, (val-1) % 4
                    distance += abs(i - target_i) + abs(j - target_j)
        return distance
    
    def is_solvable(self) -> bool:
        """Check if the current puzzle state is solvable"""
        # Flatten the puzzle (keep 0 for blank)
        flat = [num for row in self.state for num in row]
        inversions = 0
        blank_row = 0
        
        # Count inversions (excluding blank)
        for i in range(len(flat)):
            if flat[i] == 0:
                blank_row = i // 4
                continue
            for j in range(i + 1, len(flat)):
                if flat[j] == 0:
                    continue
                if flat[i] > flat[j]:
                    inversions += 1
        
        # Blank row from bottom (1-based)
        blank_row_from_bottom = 4 - blank_row
        
        # For 4x4 puzzle:
        return (inversions % 2 == 0) == (blank_row_from_bottom % 2 == 1)

def generate_puzzles(method: str, num_puzzles: int) -> List[Puzzle15]:
    puzzles = []
    
    if method == '15-puzzle':
        for _ in range(num_puzzles):
            p = Puzzle15()
            p.shuffle(moves=random.randint(10, 30))
            puzzles.append(p)
    
    elif method == 'random':
        for _ in range(num_puzzles):
            while True:
                # Generate random permutation
                tiles = list(range(16))
                random.shuffle(tiles)
                state = np.array(tiles).reshape(4,4)
                p = Puzzle15(state)
                
                # Only accept valid puzzles (unique numbers)
                if len(np.unique(p.state)) == 16:
                    puzzles.append(p)
                    break
    
    return puzzles

def evaluate_puzzles(puzzles: List[Puzzle15]) -> dict:
    results = {
        'solvable': 0,
        'avg_manhattan': 0,
        'min_manhattan': float('inf'),
        'max_manhattan': 0,
        'solve_time': 0
    }
    
    for p in puzzles:
        # Check solvability
        solvable = p.is_solvable()
        results['solvable'] += int(solvable)
        
        # Calculate heuristic
        md = p.manhattan_distance()
        results['avg_manhattan'] += md
        results['min_manhattan'] = min(results['min_manhattan'], md)
        results['max_manhattan'] = max(results['max_manhattan'], md)
        
        # Simple solving attempt (not full A* for speed)
        if solvable:
            start = time.time()
            temp_p = Puzzle15(p.state.copy())
            while not temp_p.is_solved() and (time.time() - start) < 0.1:  # 100ms timeout
                blank = temp_p.find_blank()
                # Try moves that reduce Manhattan distance
                best_move = None
                best_improvement = 0
                
                for move in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    temp_copy = Puzzle15(temp_p.state.copy())
                    if temp_copy.move(move):
                        improvement = temp_p.manhattan_distance() - temp_copy.manhattan_distance()
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = move
                
                if best_move:
                    temp_p.move(best_move)
            
            results['solve_time'] += time.time() - start
    
    results['avg_manhattan'] /= len(puzzles)
    results['solve_time'] /= max(1, results['solvable'])
    return results

if __name__ == "__main__":
    num_puzzles = 100
    methods = ['15-puzzle', 'random']
    
    for method in methods:
        print(f"\nEvaluating {method} puzzles...")
        puzzles = generate_puzzles(method, num_puzzles)
        results = evaluate_puzzles(puzzles)
        
        print(f"Solvable: {results['solvable']}/{num_puzzles} ({results['solvable']/num_puzzles*100:.1f}%)")
        print(f"Manhattan Distance - Avg: {results['avg_manhattan']:.1f}, "
              f"Min: {results['min_manhattan']}, Max: {results['max_manhattan']}")
        print(f"Avg solving time (for solvable): {results['solve_time']:.4f}s")
        
        # Example puzzle
        print("\nExample puzzle:")
        print(puzzles[0].state)


# In[ ]:




