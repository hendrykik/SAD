import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
maxsize = float('inf')
random.seed(15)


# N = int(input('Enter the size of the matrix (n x n): '))

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

def copyToFinal(curr_path):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]

def firstMin(adj, i):
    min = maxsize
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k]
    return min

def secondMin(adj, i):
    first, second = maxsize, maxsize
    for j in range(N):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]
        elif(adj[i][j] <= second and adj[i][j] != first):
            second = adj[i][j]
    return second

def TSPRec(adj, curr_bound, curr_weight, level, curr_path, visited):
    global final_res
    if level == N:
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            if curr_res < final_res:
                copyToFinal(curr_path)
                final_res = curr_res
        return
    for i in range(N):
        if adj[curr_path[level-1]][i] != 0 and visited[i] == False:
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]
            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2)
            if curr_bound + curr_weight < final_res:
                curr_path[level] = i
                visited[i] = True
                TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path, visited)
            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp
            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True

def TSP(adj):
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N
    for i in range(N):
        curr_bound += (firstMin(adj, i) + secondMin(adj, i))
    curr_bound = math.ceil(curr_bound / 2)
    visited[0] = True
    curr_path[0] = 0
    TSPRec(adj, curr_bound, 0, 1, curr_path, visited)

# adj = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
# N = 4
    
if __name__ == '__main__':
    d = {}
    for i in range(10,19):
        N = i
        adj = generate_matrix(N)
        final_path = [None] * (N + 1)
        visited = [False] * N
        final_res = maxsize

        start = time.time()
        TSP(adj)
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
