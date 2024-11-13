from datetime import datetime
import numpy as np
import random


def roulette_wheel_selection(populatie, gen=None):
    fitness_total = sum(1.0 / cromozom.fitness/cromozom.fitness for cromozom in populatie)
    punct_de_selectie = random.uniform(0, fitness_total)
    fitness_acumulat = 0.0
    for cromozom in populatie:
        fitness_acumulat += 1.0 / cromozom.fitness
        if fitness_acumulat >= punct_de_selectie:
            return cromozom


def tournament_selection(populatie, gen=None, k=40):
    turneu = random.sample(populatie, k)
    return min(turneu, key=lambda cromozom: cromozom.fitness)


selection_functions = {
    'roulette': roulette_wheel_selection,
    'tournament': tournament_selection,
}


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
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            dist = int(round(np.sqrt((coordonate[i][0] - coordonate[j][0]) ** 2 +
                                     (coordonate[i][1] - coordonate[j][1]) ** 2)))
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix


class Cromozom:
    def __init__(self, traseu=None):
        n = len(DISTANTE)
        self.traseu = traseu if traseu else random.sample(range(n), n)
        self.fitness = self.calculeaza_fitness()

    def afiseaza(self):
        lung = len(self.traseu)
        for i in range(lung-1):
            print(self.traseu[i], '-', self.traseu[i+1], ':', DISTANTE[self.traseu[i], self.traseu[i+1]])
        print(self.traseu[-1], '-', self.traseu[0], ':', DISTANTE[self.traseu[-1], self.traseu[0]])

    def calculeaza_fitness(self):
        d = DISTANTE
        lung = len(self.traseu)
        lung_traseu = sum(d[self.traseu[i], self.traseu[i + 1]] for i in range(lung - 1))
        lung_traseu += d[self.traseu[-1], self.traseu[0]]
        return lung_traseu

    def crossover(self, p2):
        lung = len(self.traseu)
        st, dr = sorted(random.sample(range(lung), 2))
        traseu_copil = [-1] * lung
        traseu_copil[st:dr] = self.traseu[st:dr]
        folosite = set(traseu_copil[st:dr])
        ramase = [oras for oras in p2.traseu[dr:] + p2.traseu[:dr] if oras not in folosite]
        pozitii_goale = list(range(dr, lung)) + list(range(st))
        for pozitie, oras in zip(pozitii_goale, ramase):
            traseu_copil[pozitie] = oras
        return Cromozom(traseu=traseu_copil)

    def mutatie(self, probab=0.01):
        lung = len(self.traseu)
        for i in range(lung):
            if random.random() < probab:
                j = random.randint(0, lung - 1)
                self.traseu[i], self.traseu[j] = self.traseu[j], self.traseu[i]
        self.fitness = self.calculeaza_fitness()

    def local_search(self, max_iterations=20):
        """Optimized 2-opt local search"""
        improved = True
        iterations = 0
        best_fitness = self.fitness
        d = DISTANTE
        min_improvement = 0.1

        while improved and iterations < max_iterations:
            improved = False
            window = min(20, len(self.traseu) // 4)

            start_points = random.sample(range(1, len(self.traseu) - 2), min(20, len(self.traseu) // 2))

            for i in start_points:
                for j in range(i + 2, i + window):
                    j = j % len(self.traseu)

                    delta = (
                        - d[self.traseu[i-1], self.traseu[i]]
                        - d[self.traseu[j], self.traseu[(j+1) % len(self.traseu)]]
                        + d[self.traseu[i-1], self.traseu[j]]
                        + d[self.traseu[i], self.traseu[(j+1) % len(self.traseu)]]
                    )

                    if delta < -min_improvement:
                        self.traseu[i:j+1] = self.traseu[i:j+1][::-1]
                        self.fitness += delta
                        improved = True
                        break

                if improved:
                    break

            iterations += 1
            if self.fitness > best_fitness * 0.99:
                break

        return self.fitness < best_fitness

    def __str__(self):
        return f"Traseu: {self.traseu}, Distanță: {self.fitness}"


def g_a(dim_pop, nr_gen, selectie):
    populatie = [Cromozom() for _ in range(dim_pop)]
    for gen in range(nr_gen):
        populatie = sorted(populatie, key=lambda x: x.fitness, reverse=False)
        urm_pop = populatie[:2]
        for _ in range(dim_pop // 2):
            p1 = selectie(populatie, gen)
            p2 = selectie(populatie, gen)
            copil1 = p1.crossover(p2)
            copil2 = p2.crossover(p1)
            copil1.mutatie()
            copil2.mutatie()
            if random.random() < 0.15:
                copil1.local_search()
            if random.random() < 0.15:
                copil2.local_search()
            urm_pop += [copil1, copil2]
        populatie = urm_pop

    best_cromozom = min(populatie, key=lambda x: x.fitness)
    return best_cromozom.traseu, best_cromozom.fitness


file_path = 'TSP/gil262.tsp'
DISTANTE = citeste_instanta_tsp(file_path)

if __name__ == '__main__':
    print(f'Numarul orașelor: {len(DISTANTE)}')
    results = []
    execution_start = datetime.now()
    for selection in selection_functions:
        start_time = datetime.now()
        best_traseu, dist = g_a(dim_pop=200, nr_gen=1000, selectie=selection_functions[selection])
        results.append((
            selection,
            dist,
            (datetime.now() - start_time).total_seconds()
        ))
        print('*')

    results.sort(key=lambda x: x[1])
    print("\nGenetic Algorithm Results")
    print("-" * 75)
    print(f"{'Selection':<12} | {'Distance':>10} | {'Time':>20}")
    print("-" * 75)

    for selection, dist, time in results:
        print(f"{selection:<12} | {dist:>10.2f} | {time:>10.2f}s")
    print("-" * 75)

    print(f"Total execution time: {datetime.now() - execution_start}")
