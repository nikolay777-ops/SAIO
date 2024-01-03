import numpy as np


def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)


# Split vertexes into two parts while checking whether it could be done
def split_graph(adjacent_matrix) -> tuple:
    colors = np.zeros(len(adjacent_matrix))
    visited = [False] * len(adjacent_matrix)

    def dfs(adjacent_matrix, vertex, previous_color):
        previous_color = 1 if previous_color == 0 else 0

        visited[vertex] = True
        colors[vertex] = previous_color

        adjacent_vertexes = np.nonzero(adjacent_matrix[vertex, :])[0]

        for vertex in adjacent_vertexes:
            if not visited[vertex]:
                dfs(adjacent_matrix, vertex, previous_color)
            elif colors[vertex] == previous_color:
                raise ValueError('Given graph is not bipartite')

    for vertex in range(len(adjacent_matrix)):
        if not visited[vertex]:
            dfs(adjacent_matrix, vertex, 0)

    return np.nonzero(colors)[0], np.nonzero(colors == 0)[0]


def in_M(M, vertex):
    for edge in M:
        if vertex in edge:
            return True

    return False


def kuhn_dfs(adjacent_matrix, vertex, target_to_source, visited, n):
    if visited[vertex]:
        return False

    visited[vertex] = True

    adjacent_vertexes = np.nonzero(adjacent_matrix[vertex, :])[0]

    for vertex_to in adjacent_vertexes:
        vertex_to -= n

        if target_to_source[vertex_to] == -1 or kuhn_dfs(adjacent_matrix, target_to_source[vertex_to], target_to_source,
                                                         visited, n):
            target_to_source[vertex_to] = vertex
            return True

    return False


def get_max_matching(adjacent_matrix, first_vertexes=None):
    if not is_symmetric(adjacent_matrix):
        raise ValueError('Adjacent matrix is not symmetric')

    if first_vertexes is None:
        first_vertexes, _ = split_graph(adjacent_matrix)
        n = len(first_vertexes)
        k = len(adjacent_matrix) - n
    else:
        n = k = int(len(adjacent_matrix) / 2)

    target_to_source = [-1] * n

    M = []
    for vertex in first_vertexes:
        visited = [False] * k

        kuhn_dfs(adjacent_matrix, vertex, target_to_source, visited, n)

    for j, i in enumerate(target_to_source):
        if i != -1:
            M.append((i, j + n))

    return M