"""
CIADA: Crow-Inspired Adaptive Displacement Algorithm
Developed by: FCandan
Domain: Industrial Digital Twin Optimization
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

class CIADA_Optimizer:
    def __init__(self, n_pebbles, search_space, target_fitness, max_iter, alpha=2.5):
        """
        Initialize the CIADA Engine.
        :param n_pebbles: Number of solution candidates (Crows' pebbles)
        :param search_space: Tuple of (Lower_Bound, Upper_Bound) as numpy arrays
        :param target_fitness: Desired accuracy threshold
        :param max_iter: Maximum generation limit (G)
        :param alpha: Convergence stiffness constant (Exp decay control)
        """
        self.n = n_pebbles
        self.L, self.U = search_space
        self.target = target_fitness
        self.G = max_iter
        self.alpha = alpha
        
        # Initialize population within physical boundary constraints
        self.dim = len(self.L)
        self.pebbles = np.random.uniform(self.L, self.U, (self.n, self.dim))
        self.best_pebble = None
        self.best_fitness = -np.inf
        self.history = []

    def solve(self, fitness_function):
        """
        Executes the optimization loop using Volumetric Displacement logic.
        """
        for t in range(self.G):
            # 1. Evaluate "Water Level" (Fitness)
            fits = np.array([fitness_function(p) for p in self.pebbles])
            
            # 2. Identify Leader Pebble (Global Best)
            current_best_idx = np.argmax(fits)
            if fits[current_best_idx] > self.best_fitness:
                self.best_fitness = fits[current_best_idx]
                self.best_pebble = self.pebbles[current_best_idx].copy()
            
            self.history.append(self.best_fitness)

            # Terminate if target is reached (Aesop's goal met)
            if self.best_fitness >= self.target:
                break

            # 3. Calculate Adaptive Volume (Delta V) - Exponential Decay
            # Metafor: Stones get smaller as crow approaches the water surface.
            delta_v = (self.U - self.L) * np.exp(-self.alpha * t / self.G)

            # 4. Displacement Operator
            new_pebbles = []
            for i in range(self.n):
                r = np.random.rand(self.dim)
                if fits[i] < self.best_fitness:
                    # Path 1: Strategic Drop (Move towards best solution)
                    new_p = self.pebbles[i] + delta_v * (self.best_pebble - self.pebbles[i]) * r
                else:
                    # Path 2: Random Gravel Search (Fine-tuning around best)
                    new_p = self.pebbles[i] + np.random.uniform(-0.1, 0.1, self.dim) * delta_v
                
                # 5. Boundary Clamping (Physical Container Walls)
                new_p = np.clip(new_p, self.L, self.U)
                new_pebbles.append(new_p)
            
            self.pebbles = np.array(new_pebbles)

        return self.best_pebble, self.best_fitness, self.history

# --- Industrial Scenario: Plant Nutrition ---
def plant_nutrition_scenario():
    # Variables: [Nitrogen (mg/kg), Water (L/day), pH]
    L_bound = np.array([0, 0, 4.5])
    U_bound = np.array([200, 10, 8.5])
    
    def fitness_eval(x):
        # Ideal Targets: N=120, W=4.5, pH=6.2
        targets = np.array([120, 4.5, 6.2])
        # Gaussian synergy model
        diff = -np.sum(((x - targets)**2) / (2 * np.array([25, 1.5, 0.4])**2))
        return 100 * np.exp(diff)

    optimizer = CIADA_Optimizer(n_pebbles=20, 
                                search_space=(L_bound, U_bound), 
                                target_fitness=99.99, 
                                max_iter=50)
    
    best_params, best_fit, history = optimizer.solve(fitness_eval)
    print(f"Optimization Finished.\nBest Params (N, W, pH): {best_params}\nFinal Yield: %{best_fit:.4f}")

if __name__ == "__main__":
    plant_nutrition_scenario()