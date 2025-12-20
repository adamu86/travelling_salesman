import time
import csv
import random
import numpy as np
from pathlib import Path
from main import read_file_tsp, distance_matrix, genetic_algorithm
import matplotlib.pyplot as plt

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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
                'convergence_threshold': 0.01,
                'max_generations': 500,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        elif experiment_type == 'standard':
            return {
                'pop_size': max(100, 2 * num_cities),
                'generations': None,
                'convergence_window': 100,
                'convergence_threshold': 0.001,
                'max_generations': 2000,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        elif experiment_type == 'thorough':
            return {
                'pop_size': max(200, 3 * num_cities),
                'generations': None,
                'convergence_window': 200,
                'convergence_threshold': 0.0005,
                'max_generations': 5000,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        
    def run_single_experiment(self, dist_matrix, config, known_optimal=None, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
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
            crossover_type=run_config.get('crossover', 'all'),
            mutation_type=run_config.get('mutation', 'all'),
            crossover_prob=run_config.get('crossover_prob', 0.9),
            mutation_prob=run_config.get('mutation_prob', 0.1),
            memetic_type=run_config.get('memetic'),
            memetic_mode=run_config.get('memetic_mode', 'all'),
            verbose=False,
            convergence_window=run_config.get('convergence_window', 100),
            convergence_threshold=run_config.get('convergence_threshold', 0.001),
            max_generations=run_config.get('max_generations', 2000)
        )
        
        result['config'] = run_config
        result['seed'] = seed
        
        if known_optimal:
            result['error_percent'] = ((result['best_length'] - known_optimal) / known_optimal) * 100
            
        return result
    
    def experiment_1_ga_combinations(self, problem_file, known_optimal=None, runs=5, experiment_type='standard'):
        print("Eksperyment 1: Wszystkie kombinacje operatorów GA")
        
        data = read_file_tsp(problem_file)
        dist_matrix = distance_matrix(data.node_coords)
        
        print(f"\nProblem: {data.name}")
        print(f"Liczba miast: {len(dist_matrix)}")
        if known_optimal:
            print(f"Znane optimum: {known_optimal}")
        
        smart_config = self.get_smart_config(len(dist_matrix), experiment_type)
        print(f"\nParametry:")
        print(f"  - Rozmiar populacji: {smart_config['pop_size']}")
        print(f"  - Max generacji: {smart_config['max_generations']}")
        print(f"  - P(krzyżowanie): {smart_config['crossover_prob']}")
        print(f"  - P(mutacja): {smart_config['mutation_prob']}")
        print(f"  - Powtórzenia: {runs}")

        crossover_ops = ['pmx', 'ox', 'erx']
        mutation_ops = ['inversion', 'scramble']

        combinations = []
        for cross in crossover_ops:
            for mut in mutation_ops:
                combinations.append({
                    'name': f'{cross.upper()}+{mut}',
                    'crossover': cross,
                    'mutation': mut
                })

        combinations.append({
            'name': 'RANDOM',
            'crossover': 'all',
            'mutation': 'all'
        })
        
        print(f"\nTestowane kombinacje:")
        for combo in combinations:
            print(f"  - {combo['name']}")
        
        results = {combo['name']: [] for combo in combinations}
        
        for combo in combinations:
            print(f"\n--- Testowanie: {combo['name']}")
            for run in range(runs):
                run_seed = RANDOM_SEED + run * 1000 + hash(combo['name']) % 1000
                print(f"  - Run {run+1}/{runs}...", end=' ', flush=True)
                
                config = {
                    'crossover': combo['crossover'],
                    'mutation': combo['mutation'],
                    'memetic': None,
                    'experiment_type': experiment_type
                }
                
                result = self.run_single_experiment(dist_matrix, config, known_optimal, seed=run_seed)
                results[combo['name']].append(result)
                
                error_str = f"(błąd: {result['error_percent']:.2f}%)" if known_optimal else ""
                print(f"Długość: {result['best_length']:.2f} w {result['generations_run']} gen {error_str}")
        
        self._save_and_plot_combinations(results, data.name, known_optimal, "exp1_ga_combinations")
        self._print_summary_table(results, "Kombinacje operatorów GA")
        return results
    
    def experiment_2_ga_vs_heuristics(self, problem_file, known_optimal=None, runs=5, experiment_type='standard'):
        print("\nEksperyment 2: GA vs Heurystyki lokalne")
        
        data = read_file_tsp(problem_file)
        dist_matrix = distance_matrix(data.node_coords)
        
        print(f"\nProblem: {data.name}")
        print(f"Liczba miast: {len(dist_matrix)}")
        if known_optimal:
            print(f"Znane optimum: {known_optimal}")
        
        from heuristics.two_opt import two_opt, route_distance
        from heuristics.three_opt import three_opt
        from heuristics.lin_kernighan_light import lin_kernighan_light
        
        results = {}

        print("\n--- Algorytm Genetyczny (bazowy, wszystkie operatory losowo)")
        ga_results = []
        for run in range(runs):
            run_seed = RANDOM_SEED + run * 1000
            print(f"  Run {run+1}/{runs}...", end=' ', flush=True)
            
            config = {
                'crossover': 'all',
                'mutation': 'all',
                'memetic': None,
                'experiment_type': experiment_type
            }
            result = self.run_single_experiment(dist_matrix, config, known_optimal, seed=run_seed)
            ga_results.append(result)
            
            error_str = f"(błąd: {result['error_percent']:.2f}%)" if known_optimal else ""
            print(f"Długość: {result['best_length']:.2f} w {result['generations_run']} gen {error_str}")
        
        results['GA'] = ga_results

        print("\n--- Heurystyka 2-opt (10 prób z losowego startu)")
        opt2_results = []
        for run in range(runs):
            run_seed = RANDOM_SEED + run * 2000
            random.seed(run_seed)
            
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
            result = {
                'best_length': best_2opt,
                'total_time': time_2opt,
                'solution': best_solution_2opt,
                'generations_run': 10,
                'convergence_history': [best_2opt]
            }
            if known_optimal:
                result['error_percent'] = ((best_2opt - known_optimal) / known_optimal) * 100
            
            opt2_results.append(result)
            error_str = f"(błąd: {result['error_percent']:.2f}%)" if known_optimal else ""
            print(f"  Run {run+1}/{runs}... Długość: {best_2opt:.2f} w {time_2opt:.2f}s {error_str}")
        
        results['2-opt'] = opt2_results

        print("\n--- Heurystyka 3-opt (10 prób z losowego startu)")
        opt3_results = []
        for run in range(runs):
            run_seed = RANDOM_SEED + run * 3000
            random.seed(run_seed)
            
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
            result = {
                'best_length': best_3opt,
                'total_time': time_3opt,
                'solution': best_solution_3opt,
                'generations_run': 10,
                'convergence_history': [best_3opt]
            }
            if known_optimal:
                result['error_percent'] = ((best_3opt - known_optimal) / known_optimal) * 100
            
            opt3_results.append(result)
            error_str = f"(błąd: {result['error_percent']:.2f}%)" if known_optimal else ""
            print(f"  Run {run+1}/{runs}... Długość: {best_3opt:.2f} w {time_3opt:.2f}s {error_str}")
        
        results['3-opt'] = opt3_results

        print("\n--- Heurystyka Lin-Kernighan (10 prób z losowego startu)")
        lk_results = []
        for run in range(runs):
            run_seed = RANDOM_SEED + run * 4000
            random.seed(run_seed)
            
            best_lk = float('inf')
            best_solution_lk = None
            start_time = time.time()
            
            for trial in range(10):
                random_solution = list(range(len(dist_matrix)))
                random.shuffle(random_solution)
                solution_lk = lin_kernighan_light(random_solution, dist_matrix, 
                                                   max_outer=5, two_opt_iters=50, three_opt_iters=10)
                dist_lk = route_distance(solution_lk, dist_matrix)
                if dist_lk < best_lk:
                    best_lk = dist_lk
                    best_solution_lk = solution_lk
            
            time_lk = time.time() - start_time
            result = {
                'best_length': best_lk,
                'total_time': time_lk,
                'solution': best_solution_lk,
                'generations_run': 10,
                'convergence_history': [best_lk]
            }
            if known_optimal:
                result['error_percent'] = ((best_lk - known_optimal) / known_optimal) * 100
            
            lk_results.append(result)
            error_str = f"(błąd: {result['error_percent']:.2f}%)" if known_optimal else ""
            print(f"  Run {run+1}/{runs}... Długość: {best_lk:.2f} w {time_lk:.2f}s {error_str}")
        
        results['LK'] = lk_results
        
        self._save_and_plot_comparison(results, data.name, known_optimal, "exp2_ga_vs_heuristics")
        self._print_summary_table(results, "GA vs Heurystyki")
        return results
    
    def experiment_3_ga_vs_memetic(self, problem_file, known_optimal=None, runs=3, experiment_type='standard'):
        print("\nEksperyment 3: GA vs Algorytmy memetyczne")
        
        data = read_file_tsp(problem_file)
        dist_matrix = distance_matrix(data.node_coords)
        
        print(f"\nProblem: {data.name}")
        print(f"Liczba miast: {len(dist_matrix)}")
        if known_optimal:
            print(f"Znane optimum: {known_optimal}")
        
        variants = [
            {'name': 'GA (bazowy)', 'memetic': None, 'mode': 'all'},
            {'name': 'MA-2opt (wszystkie)', 'memetic': '2opt', 'mode': 'all'},
            {'name': 'MA-3opt (elite 10%)', 'memetic': '3opt', 'mode': 'elite'},
            {'name': 'MA-LK (elite 10%)', 'memetic': 'lk', 'mode': 'elite'}
        ]
        
        print(f"\nTestowane warianty:")
        for v in variants:
            print(f"  - {v['name']}")
        
        results = {v['name']: [] for v in variants}
        
        for variant in variants:
            print(f"\n--- Testowanie: {variant['name']}")
            for run in range(runs):
                run_seed = RANDOM_SEED + run * 1000 + hash(variant['name']) % 1000
                print(f"  - Run {run+1}/{runs}...", end=' ', flush=True)
                
                config = {
                    'crossover': 'all',
                    'mutation': 'all',
                    'memetic': variant['memetic'],
                    'memetic_mode': variant['mode'],
                    'experiment_type': experiment_type
                }
                
                if variant['memetic']:
                    config['max_generations'] = 1000
                
                result = self.run_single_experiment(dist_matrix, config, known_optimal, seed=run_seed)
                results[variant['name']].append(result)
                
                error_str = f"(błąd: {result['error_percent']:.2f}%)" if known_optimal else ""
                print(f"Długość: {result['best_length']:.2f} w {result['generations_run']} gen, "
                      f"czas: {result['total_time']:.1f}s {error_str}")
        
        self._save_and_plot_comparison(results, data.name, known_optimal, "exp3_ga_vs_memetic")
        self._print_summary_table(results, "GA vs Algorytmy memetyczne")
        return results
    
    def _print_summary_table(self, results, title):
        print(f"\nPodsumowanie:")
        print(f"{'='*80}")
        print(f"{'Metoda':<25} {'Śr. długość':>12} {'Odch. std':>12} {'Min':>12} {'Śr. czas':>12}")
        print(f"{'-'*80}")
        
        for method, runs in results.items():
            if isinstance(runs, list) and len(runs) > 0:
                lengths = [r['best_length'] for r in runs]
                times = [r.get('total_time', 0) for r in runs]
                
                avg_length = np.mean(lengths)
                std_length = np.std(lengths)
                min_length = np.min(lengths)
                avg_time = np.mean(times)
                
                print(f"{method:<25} {avg_length:>12.2f} {std_length:>12.2f} {min_length:>12.2f} {avg_time:>12.1f}s")
        
        print(f"{'='*80}\n")
    
    def _save_and_plot_combinations(self, results, problem_name, known_optimal, filename):
        csv_file = self.output_dir / f"{filename}_{problem_name}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Combination', 'Run', 'Best_Length', 'Generations', 'Time', 'Error_%', 'Seed'])
            
            for combo_name, runs in results.items():
                for i, result in enumerate(runs):
                    error = result.get('error_percent', '')
                    seed = result.get('seed', '')
                    writer.writerow([
                        combo_name, 
                        i+1, 
                        f"{result['best_length']:.2f}",
                        result['generations_run'],
                        f"{result['total_time']:.2f}",
                        f"{error:.2f}" if error else '',
                        seed
                    ])
        
        # wykresy
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = list(results.keys())
        lengths = [[r['best_length'] for r in results[m]] for m in methods]
        times = [[r['total_time'] for r in results[m]] for m in methods]
        gens = [[r['generations_run'] for r in results[m]] for m in methods]

        # boxplot długości
        axes[0, 0].boxplot(lengths, tick_labels=methods)
        axes[0, 0].set_ylabel('Długość trasy')
        axes[0, 0].set_title(f'Jakość rozwiązań - {problem_name}')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)
        
        if known_optimal:
            axes[0, 0].axhline(y=known_optimal, color='r', linestyle='--', 
                              label='Optimum', linewidth=2)
            axes[0, 0].legend()

        # boxplot czasu
        axes[0, 1].boxplot(times, tick_labels=methods)
        axes[0, 1].set_ylabel('Czas [s]')
        axes[0, 1].set_title('Czas obliczeń')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)

        # boxplot liczby generacji
        axes[1, 0].boxplot(gens, tick_labels=methods)
        axes[1, 0].set_ylabel('Liczba generacji')
        axes[1, 0].set_title('Zbieżność')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)

        # wykres zbieżności
        for method in methods:
            max_len = max(len(r['convergence_history']) for r in results[method])
            histories = []
            for r in results[method]:
                hist = r['convergence_history'][:]
                while len(hist) < max_len:
                    hist.append(hist[-1])
                histories.append(hist)
            
            avg_history = np.mean(histories, axis=0)
            axes[1, 1].plot(avg_history, label=method, linewidth=2, alpha=0.7)
        
        if known_optimal:
            axes[1, 1].axhline(y=known_optimal, color='r', linestyle='--', 
                              label='Optimum', linewidth=2)
        
        axes[1, 1].set_xlabel('Generacja')
        axes[1, 1].set_ylabel('Długość trasy')
        axes[1, 1].set_title('Średnia zbieżność')
        axes[1, 1].legend(fontsize=8, loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_{problem_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_and_plot_comparison(self, results, problem_name, known_optimal, filename):
        csv_file = self.output_dir / f"{filename}_{problem_name}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Run', 'Best_Length', 'Generations', 'Time', 'Error_%'])
            
            for method, runs in results.items():
                for i, result in enumerate(runs):
                    error = result.get('error_percent', '')
                    writer.writerow([
                        method,
                        i+1,
                        f"{result['best_length']:.2f}",
                        result.get('generations_run', 0),
                        f"{result['total_time']:.2f}",
                        f"{error:.2f}" if error else ''
                    ])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        alg_names = list(results.keys())
        lengths = [[r['best_length'] for r in results[alg]] for alg in alg_names]
        times = [[r['total_time'] for r in results[alg]] for alg in alg_names]
        gens = [[r['generations_run'] for r in results[alg]] for alg in alg_names]

        axes[0, 0].boxplot(lengths, tick_labels=alg_names)
        axes[0, 0].set_ylabel('Długość trasy')
        axes[0, 0].set_title(f'Jakość rozwiązań - {problem_name}')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=15)
        
        if known_optimal:
            axes[0, 0].axhline(y=known_optimal, color='r', linestyle='--', 
                              label='Optimum', linewidth=2)
            axes[0, 0].legend()

        axes[0, 1].boxplot(times, tick_labels=alg_names)
        axes[0, 1].set_ylabel('Czas [s]')
        axes[0, 1].set_title('Czas obliczeń')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=15)

        axes[1, 0].boxplot(gens, tick_labels=alg_names)
        axes[1, 0].set_ylabel('Liczba generacji')
        axes[1, 0].set_title('Zbieżność')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=15)

        for alg in alg_names:
            if 'convergence_history' in results[alg][0]:
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
            axes[1, 1].axhline(y=known_optimal, color='r', linestyle='--', 
                              label='Optimum', linewidth=2)
        
        axes[1, 1].set_xlabel('Generacja')
        axes[1, 1].set_ylabel('Długość trasy')
        axes[1, 1].set_title('Średnia zbieżność')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_{problem_name}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    exp = TSPExperiment()

    experiment_type = "quick"
    
    # problem_file = "./data/coords.tsp"
    # known_optimal = 7542

    problem_file = "./data/original/eil51.tsp"
    known_optimal = 426

    # exp.experiment_1_ga_combinations(
    #     problem_file,
    #     known_optimal=known_optimal,
    #     runs=1,
    #     experiment_type=experiment_type
    # )
    #
    # exp.experiment_2_ga_vs_heuristics(
    #     problem_file,
    #     known_optimal=known_optimal,
    #     runs=1,
    #     experiment_type=experiment_type
    # )

    exp.experiment_3_ga_vs_memetic(
        problem_file,
        known_optimal=known_optimal,
        runs=1,
        experiment_type=experiment_type
    )