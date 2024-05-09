import random
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from generatory.RandomNumberGenerator import RandomNumberGenerator


def generate_tab(N, seed=12):
    generator = RandomNumberGenerator(12)

    p = []
    for i in range(N):
        p.append(generator.nextInt(1,99))

    return p


if __name__ == '__main__':
    N = 5
    p = generate_tab(N)
    print(p)

    start = time.time()
    # RS(p)
    end = time.time()

    print(end - start, "for ", N, "variables")