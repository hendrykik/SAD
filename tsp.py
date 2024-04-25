import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
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

def update_final_path(path):
    final_path[:] = path[:]
    final_path[N] = path[0]

def minimal_cost(adj_matrix, node):
    minimal_cost = float('inf')
    for j in range(N):
        if adj_matrix[node][j] < minimal_cost and j != node:
            minimal_cost = adj_matrix[node][j]
    return minimal_cost

def secondary_minimal_cost(adj_matrix, node):
    min1, min2 = float('inf'), float('inf')
    for j in range(N):
        if node != j:
            cost = adj_matrix[node][j]
            if cost < min1:
                min2 = min1
                min1 = cost
            elif cost < min2 and cost != min1:
                min2 = cost
    return min2

def TSP_recursive(adj_matrix, current_bound, current_weight, depth, path, visited_nodes):
    global final_res
    if depth == N:
        if adj_matrix[path[depth - 1]][path[0]] != 0:
            current_result = current_weight + adj_matrix[path[depth - 1]][path[0]]
            if current_result < final_res:
                update_final_path(path)
                final_res = current_result
        return

    for node in range(N):
        if adj_matrix[path[depth - 1]][node] != 0 and not visited_nodes[node]:
            temp_bound = current_bound
            current_weight += adj_matrix[path[depth - 1]][node]
            
            if depth == 1:
                current_bound -= (minimal_cost(adj_matrix, path[depth - 1]) + minimal_cost(adj_matrix, node)) / 2
            else:
                current_bound -= (secondary_minimal_cost(adj_matrix, path[depth - 1]) + minimal_cost(adj_matrix, node)) / 2
                
            if current_bound + current_weight < final_res:
                path[depth] = node
                visited_nodes[node] = True
                TSP_recursive(adj_matrix, current_bound, current_weight, depth + 1, path, visited_nodes)
                
            current_weight -= adj_matrix[path[depth - 1]][node]
            current_bound = temp_bound
            visited_nodes[node] = False

def traveling_salesman_problem(adj_matrix):
    global N, final_res, final_path
    N = len(adj_matrix)
    final_res = float('inf')
    final_path = [-1] * (N + 1)

    initial_bound = sum((minimal_cost(adj_matrix, node) + secondary_minimal_cost(adj_matrix, node)) for node in range(N))
    initial_bound = math.ceil(initial_bound / 2)

    start_node = 0
    path = [-1] * (N + 1)
    path[0] = start_node
    visited_nodes = [False] * N
    visited_nodes[start_node] = True
    
    TSP_recursive(adj_matrix, initial_bound, 0, 1, path, visited_nodes)

    
if __name__ == '__main__':
    d = {}
    for i in range(10,19):
        N = i
        adj = generate_matrix(N)
        final_path = [None] * (N + 1)
        visited = [False] * N
        final_res = maxsize

        start = time.time()
        traveling_salesman_problem(adj)
        end = time.time()

        print("Minimum cost:", final_res)
        print("Path Taken:", end=' ')
        for i in range(N + 1):
            print(final_path[i], end=' ')
        print(end - start, "for ", N, "variables")
        d[N] = end-start
    
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
