import random


def ox(parent1, parent2):
    size = len(parent1)

    child = [None] * size

    start, end = sorted(random.sample(range(size), 2))

    child[start:end] = parent1[start:end]

    rest = [x for x in parent2 if x not in child]

    index = end % size

    for element in rest:
        while child[index] is not None:
            index = (index + 1) % size

        child[index] = element

    return child