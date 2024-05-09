import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from heapq import heappop, heappush
from sys import maxsize

maxsize = float('inf')
random.seed(12)
beam_width = 10

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

def bnb_update_final_path(path):
    final_path[:] = path[:]
    final_path[N] = path[0]

def bnb_minimal_cost(adj_matrix, node):
    minimal_cost = float('inf')
    for j in range(N):
        if adj_matrix[node][j] < minimal_cost and j != node:
            minimal_cost = adj_matrix[node][j]
    return minimal_cost


def bnb_TSP_recursive(adj_matrix, current_bound, current_weight, depth, path, visited_nodes):
    global final_res
    if depth == N:
        if adj_matrix[path[depth - 1]][path[0]] != 0:
            current_result = current_weight + adj_matrix[path[depth - 1]][path[0]]
            if current_result < final_res:
                bnb_update_final_path(path)
                final_res = current_result
        return

    for node in range(N):
        if adj_matrix[path[depth - 1]][node] != 0 and not visited_nodes[node]:
            temp_bound = current_bound
            current_weight += adj_matrix[path[depth - 1]][node]

            current_bound -= (bnb_minimal_cost(adj_matrix, path[depth - 1]) + bnb_minimal_cost(adj_matrix, node)) / 2

            if current_bound + current_weight < final_res:
                path[depth] = node
                visited_nodes[node] = True
                bnb_TSP_recursive(adj_matrix, current_bound, current_weight, depth + 1, path, visited_nodes)
                
            current_weight -= adj_matrix[path[depth - 1]][node]
            current_bound = temp_bound
            visited_nodes[node] = False

def bnb_traveling_salesman_problem(adj_matrix):
    global N, final_res, final_path
    N = len(adj_matrix)
    final_res = float('inf')
    final_path = [-1] * (N + 1)

    initial_bound = math.ceil(sum((bnb_minimal_cost(adj_matrix, node)) for node in range(N)))

    start_node = 0
    path = [-1] * (N + 1)
    path[0] = start_node
    visited_nodes = [False] * N
    visited_nodes[start_node] = True
    
    bnb_TSP_recursive(adj_matrix, initial_bound, 0, 1, path, visited_nodes)

    
def bs_update_final_path(path):
    final_path[:] = path[:]
    final_path[N] = path[0]

def bs_minimal_cost(adj_matrix, node):
    minimal_cost = float('inf')
    for j in range(N):
        if adj_matrix[node][j] < minimal_cost and j != node:
            minimal_cost = adj_matrix[node][j]
    return minimal_cost


import numpy as np
import math
import time
from heapq import heappop, heappush
from sys import maxsize

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
    d1 = {}
    d2 = {}
    d1_w = {}
    d2_w = {}
    for i in range(8,14):
        N = i
        adj = generate_matrix(N)
        final_path = [None] * (N + 1)
        visited = [False] * N
        final_res = maxsize

        start = time.time()
        bnb_traveling_salesman_problem(adj)
        end = time.time()

        print("Minimum cost:", final_res)
        print("Path Taken:", end=' ')
        for i in range(N + 1):
            print(final_path[i], end=' ')
        print(end - start, "for ", N, "variables")
        d1[N] = end-start
        d1_w[N] = final_res
        

    for i in range(8,14):
        N = i
        adj = generate_matrix(N)
        final_path = [None] * (N + 1)
        visited = [False] * N
        final_res = maxsize

        start = time.time()
        bs_traveling_salesman_problem(adj)
        end = time.time()

        print("Minimum cost:", final_res)
        print("Path Taken:", end=' ')
        for i in range(N + 1):
            print(final_path[i], end=' ')
        print(end - start, "for ", N, "variables")
        d2[N] = end-start
        d2_w[N] = final_res

    

    x = list(d1.keys())
    y = list(d1.values())
    y1 = list(d2.values())

    x = [int(i) for i in x]

    x_2 = list(d1_w.keys())
    y_2 = list(d1_w.values())
    y1_2 = list(d2_w.values())

    x_2 = [int(i) for i in x_2]

    fig, axs = plt.subplots(2, 1, figsize=(8, 12))

    axs[0].plot(x, y, marker='o', linestyle='-', color='b', label='BnB')
    axs[0].plot(x, y1, marker='o', linestyle='-', color='r', label='BS')
    degree = 2
    coeffs = np.polyfit(x, y, degree)
    trend_curve = np.polyval(coeffs, x)
    axs[0].plot(x, trend_curve, linestyle='--', color='r', label='Linia trendu')
    axs[0].set_ylim(0, max(y) * 1.2)
    axs[0].set_xlabel('Wierzchołki')
    axs[0].set_ylabel('Czas')
    axs[0].set_title('Wykres zależności czasu od liczby wierzchołków z linią trendu')
    axs[0].legend()

    axs[1].plot(x_2, y_2, marker='o', linestyle='-', color='b', label='BnB')
    axs[1].plot(x_2, y1_2, marker='o', linestyle='-', color='r', label='BS')
    degree_2 = 2
    coeffs_2 = np.polyfit(x_2, y_2, degree_2)
    trend_curve_2 = np.polyval(coeffs_2, x_2)
    axs[1].plot(x_2, trend_curve_2, linestyle='--', color='r', label='Linia trendu')
    axs[1].set_ylim(0, max(y_2) * 1.2)
    axs[1].set_xlabel('Wierzchołki')
    axs[1].set_ylabel('Suma wag wierzchołków')
    axs[1].set_title('Wykres zależności sumy wag wierzchołków od liczby wierzchołków z linią trendu')
    axs[1].legend()

    plt.tight_layout()
    plt.show()