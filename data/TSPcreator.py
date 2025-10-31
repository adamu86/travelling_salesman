import random


lower = 0
upper = 500

with open('test.data', 'w', encoding='utf-8') as file:
    DIMENSION = int(input("DIMENSION: "))

    file.write(f"NAME: Test {DIMENSION}\n"
               f"TYPE: TSP\n"
               f"COMMENT: Testing for {DIMENSION} nodes\n"
               f"DIMENSION: {DIMENSION}\n"
               f"EDGE_WEIGHT: EUC_2D\n")

    file.write("NODE_COORD_SECTION\n")

    for i in range(DIMENSION):
        file.write(f"{i + 1} {random.uniform(lower, upper):.2f} {random.uniform(lower, upper):.2f}\n")

    file.write("EOF")