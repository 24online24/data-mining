from datetime import datetime
import numpy as np
import random


def one_point_crossover(self, p2):
    lung = len(self.traseu)
    punct_crossover = random.randint(2, lung - 2)
    traseu_copil = self.traseu[:punct_crossover]
    for oras in p2.traseu:
        if oras not in traseu_copil:
            traseu_copil.append(oras)
    return traseu_copil


def two_point_crossover(self, p2):
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
    return traseu_copil


def uniform_crossover(self, p2):
    lung = len(self.traseu)
    traseu_copil = [-1] * lung
    folosite = set()
    for i in range(lung):
        if random.random() < 0.5:
            oras = self.traseu[i]
        else:
            oras = p2.traseu[i]
        if oras not in folosite:
            traseu_copil[i] = oras
            folosite.add(oras)
    j = 0
    for i in range(lung):
        if traseu_copil[i] == -1:
            while p2.traseu[j] in folosite:
                j += 1
            traseu_copil[i] = p2.traseu[j]
            folosite.add(p2.traseu[j])
    return traseu_copil


def ordered_crossover(self, p2):
    lung = len(self.traseu)
    st, dr = sorted(random.sample(range(lung), 2))
    traseu_copil = [-1] * lung
    traseu_copil[st:dr] = self.traseu[st:dr]
    folosite = set(traseu_copil[st:dr])
    ramase = [oras for oras in p2.traseu[dr:] + p2.traseu[:dr] if oras not in folosite]
    pozitii_goale = list(range(dr, lung)) + list(range(st))
    for pozitie, oras in zip(pozitii_goale, ramase):
        traseu_copil[pozitie] = oras
    return traseu_copil


crossover_functions = {
    'one_point': one_point_crossover,
    'two_point': two_point_crossover,
    'uniform': uniform_crossover,
    'ordered': ordered_crossover
}


def random_selection(populatie, gen=None):
    return random.choice(populatie)


def roulette_wheel_selection(populatie, gen=None):
    fitness_total = sum(1.0 / cromozom.fitness/cromozom.fitness for cromozom in populatie)
    punct_de_selectie = random.uniform(0, fitness_total)
    fitness_acumulat = 0.0
    for cromozom in populatie:
        fitness_acumulat += 1.0 / cromozom.fitness
        if fitness_acumulat >= punct_de_selectie:
            return cromozom


def tournament_selection(populatie, gen=None, k=30):
    turneu = random.sample(populatie, k)
    return min(turneu, key=lambda cromozom: cromozom.fitness)


def rank_selection(populatie, gen=None):
    n = len(populatie)
    ranguri = [(n-i) for i in range(n)]
    suma_rangurilor = sum(ranguri)
    probabilitati = [rang / suma_rangurilor for rang in ranguri]
    punct_selectie = random.random()
    suma_cumulata = 0
    for cromozom, probabilitate in zip(populatie, probabilitati):
        suma_cumulata += probabilitate
        if suma_cumulata >= punct_selectie:
            return cromozom


def boltzmann_selection(populatie, gen):
    def normalizeaza_fitness(valori_fitness):
        min_fitness = min(valori_fitness)
        max_fitness = max(valori_fitness)
        return [1 - (f - min_fitness) / (max_fitness - min_fitness) for f in valori_fitness]

    n_gen = 100
    t_init = 10
    t_fin = 0.1
    delta = (t_init-t_fin)/n_gen
    temperatura = t_init - delta * gen
    fitness_values = [cromozom.fitness for cromozom in populatie]
    fitness_normalizat = normalizeaza_fitness(fitness_values)
    probabilitati = [np.exp(-f / temperatura) for f in fitness_normalizat]

    suma_probabilitati = sum(probabilitati)
    probabilitati = [p / suma_probabilitati for p in probabilitati]
    punct_selectie = random.random()
    suma_cumulata = 0
    for cromozom, probabilitate in zip(populatie, probabilitati):
        suma_cumulata += probabilitate
        if suma_cumulata >= punct_selectie:
            return cromozom


selection_functions = {
    'random': random_selection,
    'roulette': roulette_wheel_selection,
    'tournament': tournament_selection,
    'rank': rank_selection,
    'boltzmann': boltzmann_selection
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
    def __init__(self, crossover, traseu=None):
        n = len(DISTANTE)
        self.traseu = traseu if traseu else random.sample(range(n), n)
        self.fitness = self.calculeaza_fitness()
        self.crossover_function = crossover

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
        return Cromozom(self.crossover_function, traseu=self.crossover_function(self, p2))

    def mutatie(self, probab=0.01):
        lung = len(self.traseu)
        for i in range(lung):
            if random.random() < probab:
                j = random.randint(0, lung - 1)
                self.traseu[i], self.traseu[j] = self.traseu[j], self.traseu[i]
        self.fitness = self.calculeaza_fitness()

    def __str__(self):
        return f"Traseu: {self.traseu}, Distanță: {self.fitness}"


def g_a(dim_pop, nr_gen, selectie, crossover):
    populatie = [Cromozom(crossover) for _ in range(dim_pop)]
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
            urm_pop += [copil1, copil2]
        populatie = urm_pop

    best_cromozom = min(populatie, key=lambda x: x.fitness)
    return best_cromozom.traseu, best_cromozom.fitness


file_path = 'TSP/vm1084.tsp'
DISTANTE = citeste_instanta_tsp(file_path)

if __name__ == '__main__':
    print(f'Numarul orașelor: {len(DISTANTE)}')
    results = []
    execution_start = datetime.now()
    for selection in selection_functions:
        for crossover in crossover_functions:
            start_time = datetime.now()
            best_traseu, dist = g_a(dim_pop=100, nr_gen=500, selectie=selection_functions[selection], crossover=crossover_functions[crossover])
            results.append((
                selection,
                crossover,
                dist,
                (datetime.now() - start_time).total_seconds()
            ))

    results.sort(key=lambda x: x[2])
    print("\nGenetic Algorithm Results")
    print("-" * 75)
    print(f"{'Selection':<12} | {'Crossover':<10} | {'Distance':>10} | {'Time':>20}")
    print("-" * 75)

    for selection, crossover, dist, time in results:
        print(f"{selection:<12} | {crossover:<10} | {dist:>10.2f} | {time:>10.2f}s")
    print("-" * 75)

    print(f"Total execution time: {datetime.now() - execution_start}")
