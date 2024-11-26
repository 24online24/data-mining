import numpy as np

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

file_path = 'TSP/u724.tsp'
DISTANTE = citeste_instanta_tsp(file_path)

if __name__ == '__main__':
    traseu = [int(oras) for oras in input().split(', ')]
    lung_traseu = sum(DISTANTE[traseu[i], traseu[i + 1]] for i in range(len(traseu) - 1))
    lung_traseu += DISTANTE[traseu[-1], traseu[0]]
    print(lung_traseu)