import random
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from generatory.RandomNumberGenerator import RandomNumberGenerator


def generate_tab(n, m, seed=12):
    generator = RandomNumberGenerator(12)

    p = []
    for i in range(m):
        temp = []
        for j in range(n):
            temp.append(generator.nextInt(1,99))
        p.append(temp)

    return p

def calculate_c(n, m, p):
    # c = [[0 for i in range(n)] for j in range(m)]
    c = [0 for i in range(m)]
    kol_zad = [i for i in range(n)]


    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                c[j] = p[j][kol_zad[i]]
            elif i == 0 and j > 0:
                c[j] = c[j-1] + p[j][kol_zad[i]]
            elif i > 0 and j == 0:
                c[j] = c[j] + p[j][kol_zad[i]]
            else:
                c[j] = max(c[j-1], c[j]) + p[j][kol_zad[i]]
                
    print(c)


    # return c(m-1)


if __name__ == '__main__':
    n = 5
    m = 4
    p = generate_tab(n, m)
    print(p)

    start = time.time()
    calculate_c(n, m, p)
    end = time.time()

    print(end - start, "for ", n, "variables")