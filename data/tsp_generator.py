import random
import math
from pathlib import Path

def generate_random_tsp(num_cities, min_coord=0, max_coord=1000, seed=None):
    if seed is not None:
        random.seed(seed)
    
    coords = []
    for i in range(num_cities):
        x = random.uniform(min_coord, max_coord)
        y = random.uniform(min_coord, max_coord)
        coords.append((x, y))
    
    return coords


def save_tsp_file(coords, filename, name="Random TSP", comment=""):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"NAME: {name}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"COMMENT: {comment}\n")
        f.write(f"DIMENSION: {len(coords)}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")
        
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {x:.2f} {y:.2f}\n")
        
        f.write("EOF\n")


def generate_distance_matrix(coords):
    n = len(coords)
    dist_matrix = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist_matrix[i][j] = math.sqrt(dx * dx + dy * dy)
    
    return dist_matrix


def generate_clustered_tsp(num_cities, num_clusters=5, cluster_radius=100, cluster_spread=500, seed=None):
    if seed is not None:
        random.seed(seed)
    
    cluster_centers = []
    for _ in range(num_clusters):
        cx = random.uniform(0, cluster_spread)
        cy = random.uniform(0, cluster_spread)
        cluster_centers.append((cx, cy))
    
    coords = []
    cities_per_cluster = num_cities // num_clusters
    remaining = num_cities % num_clusters
    
    for i, (cx, cy) in enumerate(cluster_centers):
        n_cities = cities_per_cluster + (1 if i < remaining else 0)
        
        for _ in range(n_cities):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, cluster_radius)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            coords.append((x, y))
    
    return coords

if __name__ == "__main__":
    sizes = [20, 50, 100, 200]
    
    for size in sizes:
        coords_random = generate_random_tsp(size, seed=42)
        save_tsp_file(
            coords_random, 
            f"./generated/random_{size}.tsp",
            name=f"Random {size}",
            comment=f"Randomly generated {size} cities"
        )
        print(f"Wygenerowano: random_{size}.tsp")
        
        coords_clustered = generate_clustered_tsp(
            size, 
            num_clusters=max(3, size // 20),
            seed=42
        )
        save_tsp_file(
            coords_clustered,
            f"./generated/clustered_{size}.tsp",
            name=f"Clustered {size}",
            comment=f"Clustered {size} cities"
        )
        print(f"Wygenerowano: clustered_{size}.tsp")
