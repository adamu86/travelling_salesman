import json
import time
import csv
from pathlib import Path
from main import read_file_tsp, distance_matrix, genetic_algorithm
import matplotlib.pyplot as plt
import numpy as np

class TSPExperiment:
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def get_smart_config(self, num_cities, experiment_type='standard'):
        if experiment_type == 'quick':
            return {
                'pop_size': max(50, num_cities),
                'generations': None,
                'convergence_window': 50,
                'convergence_threshold': 0.01,  # 1%
                'max_generations': 500
            }
        elif experiment_type == 'standard':
            return {
                'pop_size': max(100, 2 * num_cities),
                'generations': None,
                'convergence_window': 100,
                'convergence_threshold': 0.001,  # 0.1%
                'max_generations': 2000
            }
        elif experiment_type == 'thorough':
            return {
                'pop_size': max(200, 3 * num_cities),
                'generations': None,
                'convergence_window': 200,
                'convergence_threshold': 0.0005,  # 0.05%
                'max_generations': 5000
            }
        
    def run_single_experiment(self, dist_matrix, config, known_optimal=None):
        num_cities = len(dist_matrix)
        
        base_config = self.get_smart_config(
            num_cities, 
            config.get('experiment_type', 'standard')
        )
        
        run_config = {**base_config, **config}
        
        result = genetic_algorithm(
            dist_matrix, 
            pop_size=run_config.get('pop_size'),
            generations=run_config.get('generations'),
            crossover_type=run_config['crossover'],
            mutation_type=run_config['mutation'],
            memetic_type=run_config.get('memetic'),
            memetic_mode=run_config.get('memetic_mode', 'all'),
            verbose=False,
            convergence_window=run_config.get('convergence_window', 100),
            convergence_threshold=run_config.get('convergence_threshold', 0.001),
            max_generations=run_config.get('max_generations', 2000)
        )
        
        result['config'] = run_config
        
        if known_optimal:
            result['error_percent'] = ((result['best_length'] - known_optimal) / known_optimal) * 100
            
        return result
    
    def compare_crossover_operators(self, problem_file, known_optimal=None, runs=5, experiment_type='standard'):
        print("\n" + "="*80)
        print("EKSPERYMENT 1: Porównanie operatorów krzyżowania")
        print("="*80)
        
        data = read_file_tsp(problem_file)
        dist_matrix = distance_matrix(data.node_coords)
        
        print(f"\nProblem: {data.name}")
        print(f"Liczba miast: {len(dist_matrix)}")
        if known_optimal:
            print(f"Znane optimum: {known_optimal}")
        
        smart_config = self.get_smart_config(len(dist_matrix), experiment_type)
        print(f"\nParametry:")
        print(f"  Rozmiar populacji: {smart_config['pop_size']}")
        print(f"  Max generacji: {smart_config['max_generations']}")
        print(f"  Okno zbieżności: {smart_config['convergence_window']}")
        print(f"  Próg zbieżności: {smart_config['convergence_threshold']*100}%")
        print(f"  Powtórzenia: {runs}")
        
        operators = ['pmx', 'ox', 'erx']
        results = {op: [] for op in operators}
        
        for operator in operators:
            print(f"\n--- Testowanie operatora: {operator.upper()}")
            for run in range(runs):
                print(f"  Run {run+1}/{runs}...", end=' ', flush=True)
                
                config = {
                    'crossover': operator,
                    'mutation': 'inversion',
                    'memetic': None,
                    'experiment_type': experiment_type
                }
                
                result = self.run_single_experiment(dist_matrix, config, known_optimal)
                results[operator].append(result)
                
                error_str = f"(błąd: {result['error_percent']:.2f}%)" if known_optimal else ""
                print(f"Długość: {result['best_length']:.2f} w {result['generations_run']} gen {error_str}")
        
        self._save_and_plot_crossover_comparison(results, data.name, known_optimal)
        self._print_summary_table(results, "Operatory krzyżowania")
        return results
    
    def compare_with_heuristics(self, problem_file, known_optimal=None, experiment_type='standard'):
        print("\n" + "="*80)
        print("EKSPERYMENT 2: Porównanie z heurystykami lokalnymi")
        print("="*80)
        
        data = read_file_tsp(problem_file)
        dist_matrix = distance_matrix(data.node_coords)
        
        print(f"\nProblem: {data.name}, Miasta: {len(dist_matrix)}")
        if known_optimal:
            print(f"Znane optimum: {known_optimal}")
        
        from mutation.two_opt import two_opt, route_distance
        from mutation.three_opt import three_opt
        import random
        
        results = {}
        
        # 1. Czysty GA
        print("\n--- 1. Czysty algorytm genetyczny...")
        config = {
            'crossover': 'erx',
            'mutation': 'inversion',
            'memetic': None,
            'experiment_type': experiment_type
        }
        results['GA'] = self.run_single_experiment(dist_matrix, config, known_optimal)
        print(f"   Długość: {results['GA']['best_length']:.2f} w {results['GA']['generations_run']} gen")
        
        # 2. Heurystyka 2-opt (wielokrotnie z losowego startu)
        print("\n--- 2. Heurystyka 2-opt (10 prób z losowego startu)...")
        best_2opt = float('inf')
        best_solution_2opt = None
        start_time = time.time()
        
        for trial in range(10):
            random_solution = list(range(len(dist_matrix)))
            random.shuffle(random_solution)
            solution_2opt = two_opt(random_solution, dist_matrix, max_iters=1000)
            dist_2opt = route_distance(solution_2opt, dist_matrix)
            if dist_2opt < best_2opt:
                best_2opt = dist_2opt
                best_solution_2opt = solution_2opt
        
        time_2opt = time.time() - start_time
        results['2-opt'] = {
            'best_length': best_2opt,
            'total_time': time_2opt,
            'solution': best_solution_2opt
        }
        if known_optimal:
            results['2-opt']['error_percent'] = ((best_2opt - known_optimal) / known_optimal) * 100
        
        print(f"   Długość: {best_2opt:.2f} w {time_2opt:.2f}s")
        
        # 3. Heurystyka 3-opt
        print("\n--- 3. Heurystyka 3-opt (10 prób z losowego startu)...")
        best_3opt = float('inf')
        best_solution_3opt = None
        start_time = time.time()
        
        for trial in range(10):
            random_solution = list(range(len(dist_matrix)))
            random.shuffle(random_solution)
            solution_3opt = three_opt(random_solution, dist_matrix, max_iters=100)
            dist_3opt = route_distance(solution_3opt, dist_matrix)
            if dist_3opt < best_3opt:
                best_3opt = dist_3opt
                best_solution_3opt = solution_3opt
        
        time_3opt = time.time() - start_time
        results['3-opt'] = {
            'best_length': best_3opt,
            'total_time': time_3opt,
            'solution': best_solution_3opt
        }
        if known_optimal:
            results['3-opt']['error_percent'] = ((best_3opt - known_optimal) / known_optimal) * 100
        
        print(f"   Długość: {best_3opt:.2f} w {time_3opt:.2f}s")
        
        self._save_heuristic_comparison(results, data.name, known_optimal)
        return results
    
    def compare_memetic_variants(self, problem_file, known_optimal=None, runs=3, experiment_type='standard'):
        print("\n" + "="*80)
        print("EKSPERYMENT 3: Algorytmy memetyczne")
        print("="*80)
        
        data = read_file_tsp(problem_file)
        dist_matrix = distance_matrix(data.node_coords)
        
        print(f"\nProblem: {data.name}, Miasta: {len(dist_matrix)}")
        if known_optimal:
            print(f"Znane optimum: {known_optimal}")
        
        variants = [
            {'name': 'GA', 'memetic': None},
            {'name': 'MA-2opt', 'memetic': '2opt'},
            {'name': 'MA-3opt', 'memetic': '3opt'},
            {'name': 'MA-LK', 'memetic': 'lk'}
        ]
        
        results = {v['name']: [] for v in variants}
        
        for variant in variants:
            print(f"\n▶ Testowanie: {variant['name']}")
            for run in range(runs):
                print(f"  Run {run+1}/{runs}...", end=' ', flush=True)
                
                config = {
                    'crossover': 'erx',
                    'mutation': 'inversion',
                    'memetic': variant['memetic'],
                    'memetic_mode': 'all',
                    'experiment_type': experiment_type
                }
                
                if variant['memetic']:
                    config['max_generations'] = 1000
                
                result = self.run_single_experiment(dist_matrix, config, known_optimal)
                results[variant['name']].append(result)
                
                error_str = f"(błąd: {result['error_percent']:.2f}%)" if known_optimal else ""
                print(f"Długość: {result['best_length']:.2f} w {result['generations_run']} gen, "
                      f"czas: {result['total_time']:.1f}s {error_str}")
        
        self._save_memetic_comparison(results, data.name, known_optimal)
        self._print_summary_table(results, "Algorytmy memetyczne")
        return results
    
    def _print_summary_table(self, results, title):
        """Wyświetla podsumowanie w formie tabeli"""
        print(f"\n{'='*80}")
        print(f"PODSUMOWANIE: {title}")
        print(f"{'='*80}")
        print(f"{'Metoda':<15} {'Śr. długość':>12} {'Odch. std':>12} {'Min':>12} {'Śr. czas':>12}")
        print(f"{'-'*80}")
        
        for method, runs in results.items():
            if isinstance(runs, list) and len(runs) > 0:
                lengths = [r['best_length'] for r in runs]
                times = [r.get('total_time', 0) for r in runs]
                
                avg_length = np.mean(lengths)
                std_length = np.std(lengths)
                min_length = np.min(lengths)
                avg_time = np.mean(times)
                
                print(f"{method:<15} {avg_length:>12.2f} {std_length:>12.2f} {min_length:>12.2f} {avg_time:>12.1f}s")
        
        print(f"{'='*80}\n")
    
    def _save_and_plot_crossover_comparison(self, results, problem_name, known_optimal):
        
        csv_file = self.output_dir / f"crossover_comparison_{problem_name}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Operator', 'Run', 'Best_Length', 'Generations', 'Time', 'Error_%'])
            
            for operator, runs in results.items():
                for i, result in enumerate(runs):
                    error = result.get('error_percent', '')
                    writer.writerow([
                        operator, 
                        i+1, 
                        f"{result['best_length']:.2f}",
                        result['generations_run'],
                        f"{result['total_time']:.2f}",
                        f"{error:.2f}" if error else ''
                    ])
        
        # wykresy
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        operators = list(results.keys())
        lengths = [[r['best_length'] for r in results[op]] for op in operators]
        times = [[r['total_time'] for r in results[op]] for op in operators]
        gens = [[r['generations_run'] for r in results[op]] for op in operators]
        
        # boxplot długości tras
        bp1 = axes[0, 0].boxplot(lengths, labels=[op.upper() for op in operators])
        axes[0, 0].set_ylabel('Długość trasy')
        axes[0, 0].set_title(f'Jakość rozwiązań - {problem_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        if known_optimal:
            axes[0, 0].axhline(y=known_optimal, color='r', linestyle='--', label='Optimum', linewidth=2)
            axes[0, 0].legend()
        
        # boxplot czasu
        axes[0, 1].boxplot(times, labels=[op.upper() for op in operators])
        axes[0, 1].set_ylabel('Czas [s]')
        axes[0, 1].set_title('Czas obliczeń')
        axes[0, 1].grid(True, alpha=0.3)
        
        # boxplot liczby generacji
        axes[1, 0].boxplot(gens, labels=[op.upper() for op in operators])
        axes[1, 0].set_ylabel('Liczba generacji')
        axes[1, 0].set_title('Zbieżność (liczba generacji do zatrzymania)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # wykres zbieżności (średnia)
        for op in operators:
            max_len = max(len(r['convergence_history']) for r in results[op])
            
            histories = []
            for r in results[op]:
                hist = r['convergence_history'][:]
                while len(hist) < max_len:
                    hist.append(hist[-1])
                histories.append(hist)
            
            avg_history = np.mean(histories, axis=0)
            axes[1, 1].plot(avg_history, label=op.upper(), linewidth=2, alpha=0.7)
        
        if known_optimal:
            axes[1, 1].axhline(y=known_optimal, color='r', linestyle='--', label='Optimum', linewidth=2)
        
        axes[1, 1].set_xlabel('Generacja')
        axes[1, 1].set_ylabel('Długość trasy')
        axes[1, 1].set_title('Średnia zbieżność')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"crossover_comparison_{problem_name}.png", dpi=300)
        plt.close()
        
        print(f"\nWyniki zapisano")
    
    def _save_heuristic_comparison(self, results, problem_name, known_optimal):
        
        csv_file = self.output_dir / f"heuristic_comparison_{problem_name}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Best_Length', 'Time', 'Error_%'])
            
            for method, result in results.items():
                error = ''
                if known_optimal:
                    error = f"{result.get('error_percent', 0):.2f}"
                
                time_val = result.get('total_time', 0)
                
                writer.writerow([
                    method,
                    f"{result['best_length']:.2f}",
                    f"{time_val:.2f}",
                    error
                ])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        methods = list(results.keys())
        lengths = [results[m]['best_length'] for m in methods]
        times = [results[m].get('total_time', 0) for m in methods]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        ax1.bar(methods, lengths, color=colors)
        ax1.set_ylabel('Długość trasy')
        ax1.set_title(f'Jakość rozwiązań - {problem_name}')
        ax1.grid(True, alpha=0.3, axis='y')
        
        if known_optimal:
            ax1.axhline(y=known_optimal, color='red', linestyle='--', label='Optimum', linewidth=2)
            ax1.legend()
        
        for i, (m, l) in enumerate(zip(methods, lengths)):
            ax1.text(i, l, f'{l:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.bar(methods, times, color=colors)
        ax2.set_ylabel('Czas [s]')
        ax2.set_title('Czas obliczeń')
        ax2.grid(True, alpha=0.3, axis='y')

        for i, (m, t) in enumerate(zip(methods, times)):
            ax2.text(i, t, f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"heuristic_comparison_{problem_name}.png", dpi=300)
        plt.close()
        
        print(f"\nWyniki zapisano")
    
    def _save_memetic_comparison(self, results, problem_name, known_optimal):
        
        csv_file = self.output_dir / f"memetic_comparison_{problem_name}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Algorithm', 'Run', 'Best_Length', 'Generations', 'Time', 'Error_%'])
            
            for alg_name, runs in results.items():
                for i, result in enumerate(runs):
                    error = result.get('error_percent', '')
                    writer.writerow([
                        alg_name,
                        i+1,
                        f"{result['best_length']:.2f}",
                        result['generations_run'],
                        f"{result['total_time']:.2f}",
                        f"{error:.2f}" if error else ''
                    ])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        alg_names = list(results.keys())
        lengths = [[r['best_length'] for r in results[alg]] for alg in alg_names]
        times = [[r['total_time'] for r in results[alg]] for alg in alg_names]
        gens = [[r['generations_run'] for r in results[alg]] for alg in alg_names]
        
        axes[0, 0].boxplot(lengths, labels=alg_names)
        axes[0, 0].set_ylabel('Długość trasy')
        axes[0, 0].set_title(f'Jakość rozwiązań - {problem_name}')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=15)
        if known_optimal:
            axes[0, 0].axhline(y=known_optimal, color='r', linestyle='--', label='Optimum', linewidth=2)
            axes[0, 0].legend()
        
        axes[0, 1].boxplot(times, labels=alg_names)
        axes[0, 1].set_ylabel('Czas [s]')
        axes[0, 1].set_title('Czas obliczeń')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=15)
        
        axes[1, 0].boxplot(gens, labels=alg_names)
        axes[1, 0].set_ylabel('Liczba generacji')
        axes[1, 0].set_title('Zbieżność')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=15)
        
        for alg in alg_names:
            max_len = max(len(r['convergence_history']) for r in results[alg])
            histories = []
            for r in results[alg]:
                hist = r['convergence_history'][:]
                while len(hist) < max_len:
                    hist.append(hist[-1])
                histories.append(hist)
            
            avg_history = np.mean(histories, axis=0)
            axes[1, 1].plot(avg_history, label=alg, linewidth=2, alpha=0.7)
        
        if known_optimal:
            axes[1, 1].axhline(y=known_optimal, color='r', linestyle='--', label='Optimum', linewidth=2)
        
        axes[1, 1].set_xlabel('Generacja')
        axes[1, 1].set_ylabel('Długość trasy')
        axes[1, 1].set_title('Średnia zbieżność')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"memetic_comparison_{problem_name}.png", dpi=300)
        plt.close()
        
        print(f"\nWyniki zapisano")

if __name__ == "__main__":
    exp = TSPExperiment()

    experiment_type = 'thorough'
    
    # berlin52 - znane optimum: 7542
    problem_file = "./data/coords.tsp"
    known_optimal = 7542
    
    # 1: Porównanie operatorów krzyżowania
    exp.compare_crossover_operators(
        problem_file, 
        known_optimal=known_optimal, 
        runs=5,
        experiment_type=experiment_type
    )
    
    # 2: Porównanie z heurystykami
    exp.compare_with_heuristics(
        problem_file,
        known_optimal=known_optimal,
        experiment_type=experiment_type
    )
    
    # 3: Algorytmy memetyczne
    exp.compare_memetic_variants(
        problem_file,
        known_optimal=known_optimal,
        runs=3,
        experiment_type=experiment_type
    )
    
    print(" KONIEC ")