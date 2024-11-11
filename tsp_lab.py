from datetime import datetime
import numpy as np
import random

DISTANTE = np.array([
    [0, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    [10, 0, 10, 15, 20, 25, 30, 35, 40, 45],
    [15, 10, 0, 10, 15, 20, 25, 30, 35, 40],
    [20, 15, 10, 0, 10, 15, 20, 25, 30, 35],
    [25, 20, 15, 10, 0, 10, 15, 20, 25, 30],
    [30, 25, 20, 15, 10, 0, 10, 15, 20, 25],
    [35, 30, 25, 20, 15, 10, 0, 10, 15, 20],
    [40, 35, 30, 25, 20, 15, 10, 0, 10, 15],
    [45, 40, 35, 30, 25, 20, 15, 10, 0, 10],
    [50, 45, 40, 35, 30, 25, 20, 15, 10, 0]
])


class Cromozon:
    def __init__(self, traseu=None):
        n = len(DISTANTE)
        self.traseu = traseu if traseu else random.sample(range(n), n)
        self.fitness = self.calculeaza_fitness()

    def afiseaza(self):
        lung = len(self.traseu)
        for i in range(lung-1):
            print(self.traseu[i], '-', self.traseu[i+1], ':', DISTANTE[self.traseu[i]][self.traseu[i+1]])
        print(self.traseu[-1], '-', self.traseu[0], ':', DISTANTE[self.traseu[lung-1]][self.traseu[0]])

    def calculeaza_fitness(self):
        lung = len(self.traseu)
        lung_traseu = sum([DISTANTE[self.traseu[i], self.traseu[i+1]] for i in range(lung-1)])
        lung_traseu += DISTANTE[self.traseu[lung-1], self.traseu[0]]
        return lung_traseu

    def crossover(self, p2):
        lung = len(self.traseu)
        st, dr = sorted(random.sample(range(lung), 2))
        traseu_copil = [-1] * lung
        traseu_copil[st:dr] = self.traseu[st:dr]

        folosite = set(self.traseu[st:dr])
        j = 0
        for i in range(lung):
            if traseu_copil[i] == -1:
                while p2.traseu[j] in folosite:
                    j += 1
                traseu_copil[i] = p2.traseu[j]
                folosite.add(p2.traseu[j])

        return Cromozon(traseu=traseu_copil)

    def mutatie(self, probab=0.01):
        lung = len(self.traseu)
        for i in range(lung):
            if random.random() < probab:
                j = random.randint(0, lung-1)
                self.traseu[i], self.traseu[j] = self.traseu[j], self.traseu[i]

        self.fitness = self.calculeaza_fitness()

    def __str__(self):
        return f"Traseu: {self.traseu}, Distanta: {self.fitness}"


def g_a(dim_pop, nr_gen):
    populatie = [Cromozon() for _ in range(dim_pop)]

    for gen in range(nr_gen):
        populatie = sorted(populatie, key=lambda x: x.fitness, reverse=False)
        crom = populatie[0]
        # print(f"Generatie {gen}. Cea mai buna ruta gasita: {crom.traseu} cu distanta de {crom.fitness}")

        urm_pop = populatie[:2]

        for _ in range(dim_pop // 2):
            p1, p2 = random.sample(populatie[:10], 2)
            copil1 = p1.crossover(p2)
            copil2 = p2.crossover(p1)
            copil1.mutatie()
            copil2.mutatie()
            urm_pop += [copil1, copil2]

        populatie = urm_pop

    print("-------------------------------")

    best_cromozon = min(populatie, key=lambda x: x.fitness)
    best_cromozon.afiseaza()
    return best_cromozon.traseu, best_cromozon.fitness


start_time = datetime.now()
best_crom, dist = g_a(dim_pop=1000, nr_gen=1000)
print(f"Time: {datetime.now() - start_time}")

print(f"Cea mai buna ruta gasita: {best_crom} are distanta de {dist}")
