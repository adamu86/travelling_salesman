class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Node({self.x}, {self.y})"