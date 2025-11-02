import random


def erx(parent1, parent2):
    size = len(parent1)

    edges = {gene: set() for gene in parent1}

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
    current_gene = random.choice(parent1)

    while len(child) < size:
        child.append(current_gene)

        for gene in edges:
            edges[gene].discard(current_gene)

        if edges[current_gene]:
            min_len = min(len(edges[gene]) for gene in edges[current_gene])

            candidates = [gene for gene in edges[current_gene] if len(edges[gene]) == min_len]

            current_gene = random.choice(candidates)
        else:
            unused = [gene for gene in parent1 if gene not in child]

            if unused:
                current_gene = random.choice(unused)
            else:
                break

    return child