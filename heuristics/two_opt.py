def two_opt(individual, dist_matrix, max_iters):
    best = individual
    best_distance = route_distance(best, dist_matrix)

    iters = 0
    improved = True

    while improved and iters < max_iters:
        iters += 1
        improved = False

        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue

                new_route = best[:]
                new_route[i:j] = reversed(best[i:j])
                new_distance = route_distance(new_route, dist_matrix)

                if new_distance < best_distance:
                    best = new_route
                    best_distance = new_distance
                    improved = True
                    break

            if improved:
                break

    return best


def route_distance(route, dist_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i]][route[i + 1]]
    distance += dist_matrix[route[-1]][route[0]]
    return distance