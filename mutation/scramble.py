import random


def scramble(individual, progress=None):
    n = len(individual)
    
    if progress is None:
        start, end = sorted(random.sample(range(n), 2))
    else:
        # adaptacyjna długość - maleje z postępem algorytmu
        max_length_ratio = 0.8 - (0.6 * progress)
        max_length = max(2, int(n * max_length_ratio))

        fragment_length = random.randint(2, max_length)

        start = random.randint(0, n - fragment_length)
        end = start + fragment_length - 1

    fragment = individual[start:end + 1]
    random.shuffle(fragment)
    individual[start:end + 1] = fragment
    
    return individual