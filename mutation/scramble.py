import random


def scramble(individual):
    start, end = sorted(random.sample(range(len(individual)), 2))

    fragment = individual[start:end + 1]

    random.shuffle(fragment)

    individual[start:end + 1] = fragment

    return individual