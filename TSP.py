class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Node({self.x}, {self.y})"

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