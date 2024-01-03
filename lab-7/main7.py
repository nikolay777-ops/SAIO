import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def searching_algo_BFS(graph, s, t, parent):
    visited = [False] * len(graph)
    queue = []

    queue.append(s)
    visited[s] = True

    while queue:

        u = queue.pop(0)

        for ind, val in enumerate(graph[u]):
            if visited[ind] == False and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u

    return True if visited[t] else False


def parosoch(graph, source, sink):
    parent = [-1] * len(graph)
    max_flow = 0
    steps = []

    while searching_algo_BFS(graph, source, sink, parent):

        path_flow = float("Inf")
        s = sink
        while (s != source):
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]

        max_flow += path_flow

        v = sink
        while (v != source):
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
        steps.append(parent)

    return max_flow, graph, steps


graph_2d = [[1, 1], [2, 1], [2, 2], [3, 2], [3, 3]] #np.array([
#     [1,0 ,0 ],
#     [1, 1, 0],
#     [0,1 ,1 ]
# ])
L = 3
R = 3


def make_graph2d(graph, l, r):
    N = l + r + 2
    matrix = list()
    [matrix.append([0] * N) for i in range(N)]

    matrix[0][1:l + 1] = [1] * l

    for i in range(r):
        matrix[l + i + 1][-1] = 1

    for v in graph:
        matrix[v[0]][l + v[1]] = 1

    return matrix


graph = make_graph2d(graph_2d, L, R)

par = parosoch(graph, 0, L + R + 1)
print(f'Max parosoch: {par[0]}\n')
g = par[1]

print("Parosochetaniya:\n")
for i in range(L + 1, L + R + 1):
    for j in range(1, L + 1):
        if g[i][j] == 1:
            print("[", j, ",", i - L, "]\n")
            break

G = nx.DiGraph(np.array([
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 1]
]))
nx.draw(G, with_labels=True, node_size=300, arrows=True)
plt.show()
