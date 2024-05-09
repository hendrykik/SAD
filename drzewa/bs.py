import random
import numpy as np
import math
import time
from heapq import heappop, heappush
import matplotlib.pyplot as plt
from sys import maxsize
maxsize = float('inf')
random.seed(15)

def generate_matrix(n, seed=12):
    matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            if i != j:
                row.append(random.randint(1, 30))
            else:
                row.append(0)
        matrix.append(row)

    return matrix

def bs_update_final_path(path):
    final_path[:] = path[:]
    final_path[N] = path[0]

def bs_minimal_cost(adj_matrix, node):
    minimal_cost = float('inf')
    for j in range(N):
        if adj_matrix[node][j] < minimal_cost and j != node:
            minimal_cost = adj_matrix[node][j]
    return minimal_cost



# Global variables
N = 0
final_res = maxsize
final_path = []
beam_width = 10

def bs_minimal_cost(adj_matrix, node):
    return min([cost for cost in adj_matrix[node] if cost > 0])

def bs_update_final_path(path):
    global final_path
    final_path = path + [path[0]]

def bs_TSP_recursive(adj_matrix, current_bound, current_weight, depth, path, visited_nodes):
    global final_res
    if depth == N:
        if adj_matrix[path[depth - 1]][path[0]] != 0:
            current_result = current_weight + adj_matrix[path[depth - 1]][path[0]]
            if current_result < final_res:
                final_res = current_result
                bs_update_final_path(path)
        return

    candidates = []
    for node in range(N):
        if adj_matrix[path[depth - 1]][node] != 0 and not visited_nodes[node]:
            temp_bound = current_bound
            current_weight += adj_matrix[path[depth - 1]][node]
            current_bound -= (bs_minimal_cost(adj_matrix, path[depth - 1]) + bs_minimal_cost(adj_matrix, node)) / 2
            if current_bound + current_weight < final_res:
                heappush(candidates, (current_bound + current_weight, node, path + [node], visited_nodes[:]))
            current_weight -= adj_matrix[path[depth - 1]][node]
            current_bound = temp_bound

    while candidates and len(candidates) > beam_width:
        heappop(candidates)

    while candidates:
        _, node, new_path, new_visited_nodes = heappop(candidates)
        new_visited_nodes[node] = True
        bs_TSP_recursive(adj_matrix, current_bound, current_weight + adj_matrix[new_path[-2]][node], depth + 1, new_path, new_visited_nodes)

def bs_traveling_salesman_problem(adj_matrix):
    global N, final_res, final_path
    N = len(adj_matrix)
    final_res = maxsize
    final_path = [-1] * (N + 1)
    initial_bound = math.ceil(sum(bs_minimal_cost(adj_matrix, node) for node in range(N)))
    start_node = 0
    path = [start_node]
    visited_nodes = [False] * N
    visited_nodes[start_node] = True
    bs_TSP_recursive(adj_matrix, initial_bound, 0, 1, path, visited_nodes)

if __name__ == '__main__':
    d = {}
    beam_width = 10

    for i in range(10, 19):
        N = i
        adj = generate_matrix(N)

        start = time.time()
        bs_traveling_salesman_problem(adj)
        end = time.time()

        print("Minimum cost:", final_res)
        print("Path Taken:", final_path)
        print(f"Time taken: {end - start:.4f} seconds for {N} variables")

        d[N] = end - start
    
    print(d)

    x = list(d.keys())
    y = list(d.values())

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Dane')

    degree = 2
    coeffs = np.polyfit(x, y, degree)
    trend_curve = np.polyval(coeffs, x)

    plt.plot(x, trend_curve, linestyle='--', color='r', label='Linia trendu')
    plt.ylim(0, max(y) * 1.2)

    plt.xlabel('Wierzchołki')
    plt.ylabel('Czas')
    plt.title('Wykres zależności czasu od liczby wierzchołków z linią trendu')
    plt.legend()
    plt.show()
