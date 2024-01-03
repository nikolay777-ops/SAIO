import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from kuhn8 import get_max_matching, in_M

C = np.array([
    [7, 2, 1, 9, 4],
    [9, 6, 9, 5, 5],
    [3, 8, 3, 1, 8],
    [7, 9, 4, 2, 2],
    [8, 4, 7, 4, 8]
])


def build_adjacent_matrix(edges, n):
    matrix = np.zeros((2 * n, 2 * n))

    for i, j in edges:
        j += n
        matrix[i, j] = 1
        matrix[j, i] = 1

    return matrix


def orient_graph(adjacent_matrix, max_matching):
    for i, j in zip(*np.nonzero(np.triu(adjacent_matrix))):
        if (i, j) in max_matching:
            adjacent_matrix[i, j] = 0
            adjacent_matrix[j, i] = 1
        else:
            adjacent_matrix[i, j] = 1
            adjacent_matrix[j, i] = 0


def dfs(adjacent_matrix, visited, vertex):
    if visited[vertex]:
        return

    visited[vertex] = True

    adjacent_vertexes = np.nonzero(adjacent_matrix[vertex, :])[0]

    for vertex in adjacent_vertexes:
        if not visited[vertex]:
            dfs(adjacent_matrix, visited, vertex)


def find_routes(C):
    alpha = np.zeros(C.shape[0])
    beta = np.min(C, axis=0).astype('float64')

    while True:
        J = [
            (i, j)
            for i in range(C.shape[0])
            for j in range(C.shape[1])
            if np.isclose(alpha[i] + beta[j], C[i, j])
        ]

        adjacent_matrix = build_adjacent_matrix(J, len(C))

        max_matching = get_max_matching(adjacent_matrix, first_vertexes=[i for i in range(len(C))])

        if len(max_matching) == len(C):
            return [
                (source, target - len(C)) for source, target in max_matching
            ]

        start = []
        for vertex in range(len(C)):
            if not in_M(max_matching, vertex):
                start.append(vertex)

        orient_graph(adjacent_matrix, max_matching)

        # DFS
        visited = [False] * (2 * len(C))

        for vertex in start:
            dfs(adjacent_matrix, visited, vertex)

        alpha_cap = np.where(visited[:len(C)], 1, -1)
        beta_cap = np.where(visited[len(C):], -1, 1)

        theta = np.min([
            value for value in (
                (C[i, j] - alpha[i] - beta[j]) / 2
                for i in range(len(C))
                for j in range(len(C))
                if visited[i] and not visited[j + len(C)]
            )
        ])

        alpha += theta * alpha_cap
        beta += theta * beta_cap


print(find_routes(C))

G = nx.DiGraph(C)
nx.draw(G, with_labels=True, node_size=300, arrows=True)
plt.show()
