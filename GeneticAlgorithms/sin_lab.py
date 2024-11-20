from datetime import datetime
import math
import random


class Cromozom:
    def __init__(self, numar=None):
        self.numar = numar if numar else random.uniform(-10, 10)
        self.fitness = self.calculeaza_fitness()

    def afiseaza(self):
        print(self.numar)

    def calculeaza_fitness(self):
        return self.numar * math.sin(self.numar)

    def crossover(self, p2):
        pondere = random.random()
        return Cromozom(numar=pondere*self.numar + (1-pondere)*p2.numar)

    def mutatie(self, probab=0.01, raza=0.5):
        if random.random() < probab:
            numar_mutat = self.numar + random.uniform(-raza, raza)
            self.numar = max(min(numar_mutat, 10), -10)
            self.fitness = self.calculeaza_fitness()

    def __str__(self):
        return f"Numar: {self.numar}, Valoare: {self.fitness}"


def g_a(dim_pop, nr_gen):
    populatie = [Cromozom() for _ in range(dim_pop)]

    for gen in range(nr_gen):
        populatie = sorted(populatie, key=lambda x: x.fitness, reverse=True)
        crom = populatie[0]
        # print(f"Generatie {gen}. Cel mai bun argument gasit: {crom.numar} cu distanta de {crom.fitness}")

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

    best_cromozom = max(populatie, key=lambda x: x.fitness)
    best_cromozom.afiseaza()
    return best_cromozom.numar, best_cromozom.fitness


start_time = datetime.now()
best_crom, max_val = g_a(dim_pop=1000, nr_gen=1000)
print(f"Time: {datetime.now() - start_time}")

print(f"Cea mai bun argument gasit: {best_crom} are valoarea functiei de {max_val}")
