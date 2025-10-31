import random
import matplotlib.pyplot as plt


lower = 0
upper = 500

x_vals = []
y_vals = []

with open('test.tsp', 'w', encoding='utf-8') as file:
    DIMENSION = int(input("DIMENSION: "))

    file.write(f"NAME: Test {DIMENSION}\n"
               f"TYPE: TSP\n"
               f"COMMENT: Testing for {DIMENSION} nodes\n"
               f"DIMENSION: {DIMENSION}\n"
               f"EDGE_WEIGHT: EUC_2D\n")

    file.write("NODE_COORD_SECTION\n")

    for i in range(DIMENSION):
        random1 = random.uniform(lower, upper)
        random2 = random.uniform(lower, upper)
        x_vals.append(random1)
        y_vals.append(random2)
        file.write(f"{i + 1} {random1:.2f} {random2:.2f}\n")

    file.write("EOF")

plt.scatter(x_vals, y_vals, s=40, marker='o', color='orange')
plt.title("Random nodes for test")
plt.xlabel("X")
plt.ylabel("Y")

for i, (x, y) in enumerate(zip(x_vals, y_vals)):
    plt.text(x, y, str(i), fontsize=10, color='black')

plt.savefig("../output/test_nodes.png", dpi=300)