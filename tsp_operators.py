import random
import numpy as np


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


def one_point_crossover(self, p2):
    lung = len(self.traseu)
    punct_crossover = random.randint(2, lung - 2)
    traseu_copil = self.traseu[:punct_crossover]
    for oras in p2.traseu:
        if oras not in traseu_copil:
            traseu_copil.append(oras)
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


def normalizeaza_fitness(valori_fitness):
    min_fitness = min(valori_fitness)
    max_fitness = max(valori_fitness)
    return [1 - (f - min_fitness) / (max_fitness - min_fitness) for f in valori_fitness]


def boltzmann_selection(populatie, gen):
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
