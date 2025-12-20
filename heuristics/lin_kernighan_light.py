from heuristics.two_opt import two_opt
from heuristics.three_opt import three_opt

def lin_kernighan_light(route, dist_matrix, max_outer, two_opt_iters, three_opt_iters):

    best = route[:]
    best_dist = route_distance(best, dist_matrix)

    for _ in range(max_outer):
        improved = False

        # faza 2-opt
        new_route = two_opt(best, dist_matrix, max_iters=two_opt_iters)
        new_dist = route_distance(new_route, dist_matrix)

        if new_dist < best_dist:
            best = new_route
            best_dist = new_dist
            improved = True

        # faza 3-opt
        new_route = three_opt(best, dist_matrix, max_iters=three_opt_iters)
        new_dist = route_distance(new_route, dist_matrix)

        if new_dist < best_dist:
            best = new_route
            best_dist = new_dist
            improved = True

        if not improved:
            break

    return best

def route_distance(route, dist_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i]][route[i + 1]]
    distance += dist_matrix[route[-1]][route[0]]
    return distance