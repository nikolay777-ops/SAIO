import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np


def tsp_bellman_held_karp(graph):
    num_cities = len(graph)
    # Исключаем город 0 из рассмотрения, так как начнем с него.
    num_subsets = 1 << num_cities
    dp = np.full((num_subsets, num_cities), float('inf'))
    dp[1][0] = 0  # Начинаем с города 0.
    parent = np.full((num_subsets, num_cities), -1)

    for subset in range(1, num_subsets):
        for u in range(1, num_cities):
            if (subset >> u) & 1:
                for v in range(num_cities):
                    if v != u and (subset >> v) & 1:
                        if dp[subset][u] > dp[subset ^ (1 << u)][v] + graph[v][u]:
                            dp[subset][u] = dp[subset ^ (1 << u)][v] + graph[v][u]
                            parent[subset][u] = v

    # Находим минимальный цикл, возвращаясь в город 0.
    final_subset = (1 << num_cities) - 1
    min_length = float('inf')
    last_city = -1
    for u in range(1, num_cities):
        if min_length > dp[final_subset][u] + graph[u][0]:
            min_length = dp[final_subset][u] + graph[u][0]
            last_city = u

    # Восстанавливаем маршрут, начиная с последнего города.
    tour = []
    subset = final_subset
    while last_city != -1:
        tour.append(last_city)
        new_last_city = parent[subset][last_city]
        subset ^= (1 << last_city)
        last_city = new_last_city
    tour.append(0)  # Завершаем маршрут, возвращаясь в город 0.

    return min_length, tour


graph = np.array([
    [0, 29, 20, 21],
    [29, 0, 15, 18],
    [20, 15, 0, 28],
    [21, 18, 28, 0]
])

G = nx.DiGraph(np.matrix(graph))
nx.draw(G, with_labels=True, node_size=300, arrows=True)
plt.show()

min_tour_length, tour_path = tsp_bellman_held_karp(graph)
print("Минимальная длина маршрута:", min_tour_length)
print("Маршрут по городам:", tour_path)