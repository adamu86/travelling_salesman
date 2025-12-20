from matplotlib import pyplot as plt
from model.Node import Node
from pathlib import Path


class TSP:
    def __init__(self, name=None, type_=None, comment=None, dimension=None, edge_weight_type=None):
        self.name = name
        self.type = type_
        self.comment = comment
        self.dimension = dimension
        self.edge_weight_type = edge_weight_type
        self.node_coords = {}

    def add_node(self, node_id, x, y):
        self.node_coords[node_id] = Node(x, y)

    def __repr__(self):
        return (f"TSP("
                f"name={self.name}, "
                f"type={self.type}, "
                f"comment={self.comment}, "
                f"dimension={self.dimension}, "
                f"edge_weight_type={self.edge_weight_type}, "
                f"nodes_coords={self.node_coords})")

    def plot(self, best_solution=None):
        plt.clf()
        plt.figure(figsize=(9, 7))

        if best_solution is None:
            x_route = [self.node_coords[i].x for i in self.node_coords]
            y_route = [self.node_coords[i].y for i in self.node_coords]

            plt.scatter(x_route, y_route, s=30, marker='o', color='orange')
            plt.title(f"Nodes for: {self.name}")

            for i, (x, y) in enumerate(zip(x_route, y_route)):
                plt.text(x, y, str(i), fontsize=8, color='black')

            Path("./output").mkdir(parents=True, exist_ok=True)

            plt.savefig(f"./output/{self.name}_nodes.png")

            return

        x_route = [self.node_coords[i].x for i in best_solution]
        y_route = [self.node_coords[i].y for i in best_solution]

        x_route.append(x_route[0])
        y_route.append(y_route[0])

        plt.scatter([node.x for node in self.node_coords.values()],
                    [node.y for node in self.node_coords.values()],
                    s=30, marker='o', color='blue')

        plt.plot(x_route, y_route, color='orange', linewidth=1, marker='o')

        for key, node in self.node_coords.items():
            if key == best_solution[0]:
                plt.text(node.x, node.y, str(key), fontsize=15, color='red')
                plt.text(node.x - 8, node.y - 20, 'START', fontsize=10, color='red')
            elif key == best_solution[-1]:
                plt.text(node.x, node.y, str(key), fontsize=15, color='green')
                plt.text(node.x - 6, node.y - 20, 'END', fontsize=10, color='green')
            else:
                plt.text(node.x, node.y, str(key), fontsize=8, color='black')

        Path("./output").mkdir(parents=True, exist_ok=True)

        plt.title(f"Best route for: {self.name}")
        plt.savefig(f"./output/{self.name}_result.png")

