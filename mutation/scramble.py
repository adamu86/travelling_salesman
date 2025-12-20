import random


def scramble(individual, progress=None):
    n = len(individual)
    
    if n < 2:
        return individual
    
    if progress is None:
        start, end = sorted(random.sample(range(n), 2))
        if end - start < 1:
            end = min(start + 2, n - 1)
    else:
        # adaptacyjna długość - maleje z postępem algorytmu
        max_length_ratio = max(0.2, 0.8 - (0.6 * progress))
        max_length = max(2, int(n * max_length_ratio))

        fragment_length = random.randint(2, min(max_length, n))

        max_start = n - fragment_length
        start = random.randint(0, max(0, max_start))
        end = start + fragment_length - 1

    fragment = individual[start:end + 1]
    
    if len(fragment) > 1:
        random.shuffle(fragment)
        individual[start:end + 1] = fragment
    
    return individual