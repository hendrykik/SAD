import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from generatory.RandomNumberGenerator import RandomNumberGenerator

# Generacja danych
def generate_instance(n, seed):
    generator = RandomNumberGenerator(seed)
    np.random.seed(seed)
    pij = generator.nextInt(1,99)
    A = np.sum(pij) // 6
    B = np.sum(pij) // 2
    dj = np.random.randint(A, B, size=n)
    return pij, dj

# Funkcja harmonogramu
def calculate_schedule(pij, perm):
    n = len(perm)
    C = np.zeros((3, n), dtype=int)
    for j in range(n):
        job = perm[j]
        for i in range(3):
            if i == 0:
                if j == 0:
                    C[i][j] = pij[i][job]
                else:
                    C[i][j] = C[i][j-1] + pij[i][job]
            else:
                if j == 0:
                    C[i][j] = C[i-1][j] + pij[i][job]
                else:
                    C[i][j] = max(C[i][j-1], C[i-1][j]) + pij[i][job]
    return C

# Funkcja kryteriów
def calculate_criteria(C, dj):
    total_flowtime = np.sum(C[2])
    total_tardiness = np.sum(np.maximum(0, C[2] - dj))
    max_lateness = np.max(C[2] - dj)
    total_lateness = np.sum(C[2] - dj)
    return total_flowtime, total_tardiness, max_lateness, total_lateness

# Funkcja sąsiedztwa (insert)
def get_neighbor(perm):
    a, b = np.random.choice(len(perm), 2, replace=False)
    perm[a], perm[b] = perm[b], perm[a]
    return perm

# Algorytm symulowanego wyżarzania
def simulated_annealing(pij, dj, max_iter, prob=0.1):
    n = len(dj)
    perm = list(np.random.permutation(n))
    C = calculate_schedule(pij, perm)
    current_criteria = calculate_criteria(C, dj)
    P = [perm]
    for it in range(max_iter):
        new_perm = get_neighbor(perm.copy())
        new_C = calculate_schedule(pij, new_perm)
        new_criteria = calculate_criteria(new_C, dj)
        if new_criteria < current_criteria:
            perm = new_perm
            current_criteria = new_criteria
        else:
            if np.random.rand() < prob:
                perm = new_perm
                current_criteria = new_criteria
        P.append(perm)
    return P, perm

# Wizualizacja wyników
def plot_pareto(P, pij, dj):
    front = []
    for perm in P:
        C = calculate_schedule(pij, perm)
        criteria = calculate_criteria(C, dj)
        front.append(criteria[:2])  # Only plot the first two criteria
    front = np.array(front)
    plt.scatter(front[:, 0], front[:, 1])
    plt.xlabel('Total Flowtime')
    plt.ylabel('Total Tardiness')
    plt.title('Pareto Front')
    plt.show()

# Przykład użycia
n = 10  # liczba zadań
seed = 42
max_iter = 1000

pij, dj = generate_instance(n, seed)
P, final_perm = simulated_annealing(pij, dj, max_iter)
plot_pareto(P, pij, dj)
