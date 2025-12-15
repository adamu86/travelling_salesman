def three_opt(route, dist_matrix, max_iters=5):
    best = route[:]
    best_dist = route_distance(best, dist_matrix)
    n = len(route)

    iters = 0
    improved = True

    # pętla ulepszania (algorytm hill climbing)
    while improved and iters < max_iters:
        iters += 1
        improved = False

        for i in range(1, n - 4):
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n):

                    A = best[:i]
                    B = best[i:j]
                    C = best[j:k]
                    D = best[k:]

                    # lista kandydatów do sprawdzenia
                    candidates = [
                        A + B + C + D,
                        A + B[::-1] + C + D,
                        A + B + C[::-1] + D,
                        A + C + B + D,
                        A + C[::-1] + B + D,
                        A + C + B[::-1] + D,
                        A + C[::-1] + B[::-1] + D,
                        A + B[::-1] + C[::-1] + D,
                    ]

                    for cand in candidates:
                        d = route_distance(cand, dist_matrix)
                        if d < best_dist:
                            best = cand
                            best_dist = d
                            improved = True
                            break
                    if improved:
                        break
                if improved:
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
