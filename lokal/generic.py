import random
import time
import os
import sys
from typing import List, Tuple

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from generatory.RandomNumberGenerator import RandomNumberGenerator

class Job:
    def __init__(self, processing_time: int, due_date: int, weight: int):
        self.processing_time = processing_time
        self.due_date = due_date
        self.weight = weight

def generator(n: int, seed: int = 12) -> List[Job]:
    rng = RandomNumberGenerator(seed)
    jobs = []
    for _ in range(n):
        processing_time = rng.nextInt(1, 99)
        due_date = rng.nextInt(1, 99)
        weight = rng.nextInt(1, 99)
        jobs.append(Job(processing_time, due_date, weight))
    return jobs

def fitness(schedule: List[Job]) -> int:
    current_time = 0
    total_weighted_tardiness = 0
    for job in schedule:
        current_time += job.processing_time
        tardiness = max(0, current_time - job.due_date)
        total_weighted_tardiness += job.weight * tardiness
    return total_weighted_tardiness

def initialize_population(jobs: List[Job], population_size: int) -> List[List[Job]]:
    population = []
    for _ in range(population_size):
        individual = jobs[:]
        random.shuffle(individual)
        population.append(individual)
    return population

def select_parents(population: List[List[Job]]) -> Tuple[List[Job], List[Job]]:
    return random.choices(population, k=2)

def crossover(parent1: List[Job], parent2: List[Job]) -> List[Job]:
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    remaining_jobs = [job for job in parent2 if job not in child]
    for i in range(size):
        if child[i] is None:
            child[i] = remaining_jobs.pop(0)
    return child

def mutate(schedule: List[Job], mutation_rate: float = 0.01):
    for i in range(len(schedule)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(schedule) - 1)
            schedule[i], schedule[j] = schedule[j], schedule[i]

def genetic_algorithm(jobs: List[Job], population_size: int, generations: int, mutation_rate: float) -> List[Job]:
    population = initialize_population(jobs, population_size)
    best_schedule = min(population, key=fitness)
    best_fitness = fitness(best_schedule)

    for generation in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        current_best_schedule = min(population, key=fitness)
        current_best_fitness = fitness(current_best_schedule)
        
        if current_best_fitness < best_fitness:
            best_schedule = current_best_schedule
            best_fitness = current_best_fitness
        
        print(f'Generation {generation + 1}, Best Fitness: {best_fitness}')

    return best_schedule

# Parameters
n = 10  # Number of jobs
seed = 12
population_size = 50
generations = 100
mutation_rate = 0.01

# Generate jobs
jobs = generator(n, seed)

# Run genetic algorithm
best_schedule = genetic_algorithm(jobs, population_size, generations, mutation_rate)
print("Best schedule found:")
for job in best_schedule:
    print(f'Processing Time: {job.processing_time}, Due Date: {job.due_date}, Weight: {job.weight}')
