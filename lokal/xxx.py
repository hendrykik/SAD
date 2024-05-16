import random
import time
import os
import sys

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


def random_search(n, m, p, iterations=1000):
    current_solution = list(range(n))
    random.shuffle(current_solution)
    current_makespan = calculate_makespan(n, m, p, current_solution)

    for _ in range(iterations):
        neighbor_solution = get_neighbor(current_solution)
        neighbor_makespan = calculate_makespan(n, m, p, neighbor_solution)

        if neighbor_makespan < current_makespan:
            current_solution = neighbor_solution
            current_makespan = neighbor_makespan

    return current_solution, current_makespan


if __name__ == '__main__':
    n = 5
    m = 4
    p = generate_tab(n, m)
    print("Processing times matrix (p):")
    for row in p:
        print(row)

    # Random Search
    start = time.time()
    best_perm_rs, best_makespan_rs = random_search(n, m, p)
    end = time.time()
    print("\nRandom Search:")
    print("Best permutation:", best_perm_rs)
    print("Best makespan:", best_makespan_rs)
    print("Execution time:", end - start, "seconds")
