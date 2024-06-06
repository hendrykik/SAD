import random
import time
import os
import sys
import matplotlib.pyplot as plt
import numpy
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


def random_search(n, m, p, iterations=100000):
    current_solution = list(range(n))
    random.shuffle(current_solution)
    current_makespan = calculate_makespan(n, m, p, current_solution)

    for _ in range(iterations):
        neighbor_solution = get_neighbor(current_solution)
        neighbor_makespan = calculate_makespan(n, m, p, neighbor_solution)

        if neighbor_makespan < current_makespan:
            current_solution = neighbor_solution
            current_makespan = neighbor_makespan
            print(_, current_makespan)


    return current_solution, current_makespan


def simulated_annealing(n, m, p, initial_temp=1000, cooling_rate=0.999, stopping_temp=1, max_iterations=100000):
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
            print(iteration, best_makespan)

        temp *= cooling_rate
        iteration += 1

    print(best_solution)
    return best_solution, best_makespan



if __name__ == '__main__':
    ns = [5, 10, 15, 18, 20, 25]
    m = 10

    rs_times = []
    sa_times = []

    for n in ns:
        p = generate_tab(n, m)

        # Random Search
        start = time.time()
        best_perm_rs, best_makespan_rs = random_search(n, m, p)
        print(best_perm_rs, best_makespan_rs)
        end = time.time()
        rs_times.append(end - start)

        # Simulated Annealing
        start = time.time()
        best_perm_sa, best_makespan_sa = simulated_annealing(n, m, p)
        print(best_perm_sa, best_makespan_sa)
        end = time.time()
        sa_times.append(end - start)


    print("Czasy RS", rs_times)
    print("Czasy SA", sa_times)
    # Wykres porównujący czasy wykonania dla Random Search i Simulated Annealing
    plt.figure(figsize=(10, 6))
    plt.plot(ns, rs_times, marker='o', linestyle='-', label='Random Search')
    plt.plot(ns, sa_times, marker='o', linestyle='-', label='Simulated Annealing')
    plt.xlabel('Liczba problemów (n)')
    plt.ylabel('Czas wykonania (s)')
    plt.title('Porównanie czasów wykonania dla Random Search i Simulated Annealing')
    plt.legend()
    plt.grid(True)
    plt.show()

