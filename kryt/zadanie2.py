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



def scalarize_criteria(crit, weights):
    return sum(w * c for w, c in zip(weights, crit))

def simulated_annealing_scalarization(pij, dj, max_iter, weights, prob=0.1):
    n = len(dj)
    perm = list(np.random.permutation(n))
    C = calculate_schedule(pij, perm)
    current_criteria = calculate_criteria(C, dj)
    current_score = scalarize_criteria(current_criteria, weights)
    P = [perm]
    for it in range(max_iter):
        new_perm = get_neighbor(perm.copy())
        new_C = calculate_schedule(pij, new_perm)
        new_criteria = calculate_criteria(new_C, dj)
        new_score = scalarize_criteria(new_criteria, weights)
        if new_score < current_score:
            perm = new_perm
            current_score = new_score
        else:
            if np.random.rand() < prob:
                perm = new_perm
                current_score = new_score
        P.append(perm)
    return P, perm

# Przykład użycia

# Przykład użycia
n = 10  # liczba zadań
seed = 42
max_iter = 1000

weights = [1, 1, 1, 1]  # wagi dla kryteriów
pij, dj = generate_instance(n, seed)
P, final_perm = simulated_annealing_scalarization(pij, dj, max_iter, weights)
plot_pareto(P, pij, dj)

def plot_multiway_dotplots(P, pij, dj):
    fig, ax = plt.subplots()
    for perm in P:
        C = calculate_schedule(pij, perm)
        criteria = calculate_criteria(C, dj)
        ax.plot(criteria, 'o-')
    plt.xlabel('Kryterium')
    plt.ylabel('Wartość')
    plt.title('Wykres kropkowy (Multiway Dotplots)')
    plt.show()

def plot_star_coordinate_system(P, pij, dj):
    from math import pi
    categories = ['Total Flowtime', 'Total Tardiness', 'Max Lateness', 'Total Lateness']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for perm in P:
        C = calculate_schedule(pij, perm)
        criteria = calculate_criteria(C, dj)
        values = list(criteria)
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Perm: {perm}')
        ax.fill(angles, values, alpha=0.25)
    
    plt.xticks(angles[:-1], categories)
    plt.title('Współrzędne gwiazdowe (Star Coordinate System)')
    plt.legend()
    plt.show()

def plot_star_coordinate_system_with_lines(P, pij, dj):
    from math import pi
    categories = ['Total Flowtime', 'Total Tardiness', 'Max Lateness', 'Total Lateness']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for perm in P:
        C = calculate_schedule(pij, perm)
        criteria = calculate_criteria(C, dj)
        values = list(criteria)
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2)
    
    plt.xticks(angles[:-1], categories)
    plt.title('Odcinkowe współrzędne gwiazdowe (Star Coordinate System with Lines)')
    plt.show()

def plot_spider_web_charts(P, pij, dj):
    from math import pi
    categories = ['Total Flowtime', 'Total Tardiness', 'Max Lateness', 'Total Lateness']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for perm in P:
        C = calculate_schedule(pij, perm)
        criteria = calculate_criteria(C, dj)
        values = list(criteria)
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
    
    plt.xticks(angles[:-1], categories)
    plt.title('Wykres pajęczynowy (Spider Web Charts)')
    plt.show()

# Przykład użycia wizualizacji
plot_multiway_dotplots(P, pij, dj)
plot_star_coordinate_system(P, pij, dj)
plot_star_coordinate_system_with_lines(P, pij, dj)
plot_spider_web_charts(P, pij, dj)

