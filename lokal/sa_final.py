import random
import time
import os
import sys
import math

# Assuming RandomNumberGenerator is correctly implemented in the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from generatory.RandomNumberGenerator import RandomNumberGenerator


def generate_tab(n, m, seed=12):
    generator = RandomNumberGenerator(seed)

    p = []
    for i in range(m):
        temp = []
        for j in range(n):
            temp.append(generator.nextInt(1, 99))
        p.append(temp)

    return p


def calculate_makespan(n, m, p, kol_zad):
    c = [[0 for _ in range(n)] for _ in range(m)]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                c[j][i] = p[j][kol_zad[i]]
            elif i == 0:
                c[j][i] = c[j-1][i] + p[j][kol_zad[i]]
            elif j == 0:
                c[j][i] = c[j][i-1] + p[j][kol_zad[i]]
            else:
                c[j][i] = max(c[j-1][i], c[j][i-1]) + p[j][kol_zad[i]]

    return c[m-1][n-1]


def get_neighbor(permutation):
    neighbor = permutation[:]
    i, j = random.sample(range(len(permutation)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def simulated_annealing(n, m, p, initial_temp=1000, cooling_rate=0.99, stopping_temp=1, max_iterations=100000):
    current_solution = list(range(n))
    random.shuffle(current_solution)
    current_makespan = calculate_makespan(n, m, p, current_solution)
    temp = initial_temp

    best_solution = current_solution[:]
    best_makespan = current_makespan

    iteration = 0

    while temp > stopping_temp and iteration < max_iterations:
        neighbor_solution = get_neighbor(current_solution)
        neighbor_makespan = calculate_makespan(n, m, p, neighbor_solution)

        if neighbor_makespan < current_makespan:
            current_solution = neighbor_solution
            current_makespan = neighbor_makespan
        else:
            acceptance_prob = math.exp((current_makespan - neighbor_makespan) / temp)
            if random.uniform(0, 1) < acceptance_prob:
                current_solution = neighbor_solution
                current_makespan = neighbor_makespan

        if current_makespan < best_makespan:
            best_solution = current_solution[:]
            best_makespan = current_makespan

        temp *= cooling_rate
        iteration += 1

    return best_solution, best_makespan


if __name__ == '__main__':
    n = 100
    m = 10
    p = generate_tab(n, m)
    print("Processing times matrix (p):")
    for row in p:
        print(row)

    start = time.time()
    best_perm_rs, best_makespan_rs = simulated_annealing(n, m, p)
    end = time.time()
    print("\nRandom Search:")
    print("Best permutation:", best_perm_rs)
    print("Best makespan:", best_makespan_rs)
    print("Execution time:", end - start, "seconds")
