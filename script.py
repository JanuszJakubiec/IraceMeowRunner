import random

def generate_graph(n, m):
    edges = []
    final = []
    for i in range(1, n):
        for j in range(i+1, n+1):
            edges.append([i, j])
    for i in range(m):
        index = random.randint(0, len(edges)-1)
        final.append(edges[index])
        del edges[index]
    print(final)


generate_graph(9, 30)
