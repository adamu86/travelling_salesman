# Algorytm genetyczny dla problemu komiwojażera

>Algorytm genetyczny jako narzędzie do rozwiązywania problemu komiwojażera. Implementacja algorytmu genetycznego opartego na reprezentacji permutacyjnej do rozwiązywania problemu komiwojażera. Algorytm powinien wykorzystywać 3 różne operatory krzyżowania opisane w literaturze i operator mutacji. Badania eksperymentalne: a) porównanie ze standardowymi benchmarkami dla problemu komiwojażera (np. biblioteka TSPLIB). b) własny generator grafów losowych dla problemu komiwojażera i porównanie ze standardową heurystyką (np. Lina-Kernighana, 3-opt, 2-opt). Algorytm memetyczny, w którym każde nowe rozwiązanie wygenerowane przez operatory krzyżowania i mutacji jest ulepszane przez algorytm optymalizacji lokalnej.

## Punktacja

#### Minimum:

- ~~`[25 pkt]` Algorytm z dwoma operatorami krzyżowania i czytający dane wejściowe w formacie TSPLIB.~~ ✔️

---

#### Dodatkowo:

- ~~`[5 pkt]` trzeci operator krzyżowania 5pkt.~~ ✔️

- `[10 pkt]` własny generator znaków losowych i porównanie z heursytyką 2-opt. 

- ~~`[10 pkt]` algorytm memetyczny.~~ ✔️

---

#### Bonusowo:

- ~~`[10 pkt]` algorytm memetyczny wykorzystuje heurystykę 3-opt~~ ✔️

- `[15 pkt]` algorytm memetyczny wykorzystuje heurystykę LK

--- 

##### Dane:
- [berlin52](https://www.kaggle.com/datasets/keknure/berlin52)
