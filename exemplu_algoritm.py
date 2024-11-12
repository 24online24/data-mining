import random
import numpy as np
from exemplu_operatori import one_point_crossover as crossover

from exemplu_operatori import roulette_wheel_selection as selectie


def citeste_instanta_tsp(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    start_coord = False
    coordonate = []
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            start_coord = True
            continue
        elif "EOF" in line:
            break
        if start_coord:
            parts = line.strip().split()
            if len(parts) == 3:
                _, x, y = parts
                coordonate.append((float(x), float(y)))
    n = len(coordonate)
    print(f'Numarul orașelor: {n}')
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            dist = int(round(np.sqrt((coordonate[i][0] - coordonate[j][0]) ** 2 +
                                     (coordonate[i][1] - coordonate[j][1]) ** 2)))
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix


file_path = 'd1291.tsp'

dist = citeste_instanta_tsp(file_path)


class Cromozom:
    def __init__(self, traseu=None):
        n = len(dist)
        self.traseu = traseu if traseu else random.sample(range(n), n)
        self.fitness = self.calculeaza_fitness()

    def afiseaza(self):
        lung = len(self.traseu)
        for i in range(lung-1):
            print(self.traseu[i], '-', self.traseu[i+1], ':', dist[self.traseu[i], self.traseu[i+1]])
        print(self.traseu[-1], '-', self.traseu[0], ':', dist[self.traseu[-1], self.traseu[0]])

    def calculeaza_fitness(self):
        lung = len(self.traseu)
        lung_traseu = sum(dist[self.traseu[i], self.traseu[i + 1]] for i in range(lung - 1))
        lung_traseu += dist[self.traseu[-1], self.traseu[0]]
        return lung_traseu

    def crossover(self, p2):
        return Cromozom(traseu=crossover(self, p2))

    def mutatie(self, probab=0.01):
        lung = len(self.traseu)
        for i in range(lung):
            if random.random() < probab:
                j = random.randint(0, lung - 1)
                self.traseu[i], self.traseu[j] = self.traseu[j], self.traseu[i]
        self.fitness = self.calculeaza_fitness()

    def __str__(self):
        return f"Traseu: {self.traseu}, Distanță: {self.fitness}"


def g_a(dim_pop, nr_gen):
    populatie = [Cromozom() for _ in range(dim_pop)]
    for gen in range(nr_gen):
        populatie = sorted(populatie, key=lambda x: x.fitness, reverse=False)
        crom = populatie[0]
        print(f"Generatie {gen} cea mai bună rută are distanța {crom.fitness}")
        urm_pop = populatie[:2]
        for _ in range(dim_pop // 2):
            p1 = selectie(populatie, gen)
            p2 = selectie(populatie, gen)
            copil1 = p1.crossover(p2)
            copil2 = p2.crossover(p1)
            copil1.mutatie()
            copil2.mutatie()
            urm_pop += [copil1, copil2]
        populatie = urm_pop
    print('--------------------------')

    best_cromozom = min(populatie, key=lambda x: x.fitness)
    return best_cromozom.traseu, best_cromozom.fitness


best_traseu, dist = g_a(dim_pop=100, nr_gen=500)
print(f"Cea mai bună rută găsită: {best_traseu} \nare distanța {dist}")
