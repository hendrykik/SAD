import random
import time
import os
import sys
import matplotlib.pyplot as plt
import numpy

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

    return current_solution, current_makespan


if __name__ == '__main__':
    ns = [5, 10, 15, 20]
    ms = [5, 10, 15, 20]
    times = []

    for n in ns:
        for m in ms:
            p = generate_tab(n, m)

            start = time.time()
            best_perm_rs, best_makespan_rs = random_search(n, m, p)
            end = time.time()

            times.append((n, m, end - start))

    plt.figure(figsize=(10, 6))
    for n in ns:
        n_times = [time for n_val, m_val, time in times if n_val == n]
        plt.plot(ms, n_times, label=f'n={n}')
    plt.xlabel('Liczba maszyn (m)')
    plt.ylabel('Czas wykonania (s)')
    plt.title('Czas wykonania w zależności od liczby maszyn')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for m in ms:
        m_times = [time for n_val, m_val, time in times if m_val == m]
        plt.plot(ns, m_times, label=f'm={m}')
    plt.xlabel('Liczba problemów (n)')
    plt.ylabel('Czas wykonania (s)')
    plt.title('Czas wykonania w zależności od liczby problemów')
    plt.legend()
    plt.grid(True)
    plt.show()


