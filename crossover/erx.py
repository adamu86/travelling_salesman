import random


def ex(parent1, parent2):
    size = len(parent1)

    edges = {}
    for city in parent1:
        edges[city] = set()

    for i in range(size):
        city = parent1[i]
        left = parent1[(i - 1) % size]
        right = parent1[(i + 1) % size]
        edges[city].update([left, right])

        city = parent2[i]
        left = parent2[(i - 1) % size]
        right = parent2[(i + 1) % size]
        edges[city].update([left, right])

    child = []
    current = random.choice(parent1)

    while len(child) < size:
        child.append(current)

        for city in edges:
            edges[city].discard(current)

        if not edges[current]:
            unused = [c for c in parent1 if c not in child]

            if unused:
                current = random.choice(unused)
            continue

        current = min(edges[current], key=lambda x: len(edges[x]))

    return child