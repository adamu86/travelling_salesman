import random


def erx(parent1, parent2):
    size = len(parent1)

    neighbors = {gene: set() for gene in parent1}

    for i in range(size):
        neighbors[parent1[i]].update([parent1[(i - 1) % size], parent1[(i + 1) % size]])

        neighbors[parent2[i]].update([parent2[(i - 1) % size], parent2[(i + 1) % size]])

    child = []

    current_gene = random.choice(parent1)

    while len(child) < size:
        child.append(current_gene)

        for gene in neighbors:
            neighbors[gene].discard(current_gene)

        if neighbors[current_gene]:
            next_gene = min(neighbors[current_gene], key=lambda x: len(neighbors[x]))
        else:
            remaining_genes = [gene for gene in parent1 if gene not in child]

            if remaining_genes:
                next_gene = random.choice(remaining_genes)
            else:
                break

        current_gene = next_gene

    return child