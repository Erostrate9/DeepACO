import torch
import time

class GeneticAlgorithmTSP:
    def __init__(self, 
                 distances, 
                 pop_size=50,
                 crossover_rate=0.8,
                 mutation_rate=0.2,
                 elitism_count=1,
                 tournament_size=5,
                 device='cpu'):
        
        self.distances = distances
        self.problem_size = distances.shape[0]
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.device = device
        
        # Initialize population: a set of permutations (tours)
        self.population = self._init_population()
        # Calculate fitness for initial population
        self.fitness, self.costs = self._evaluate_population(self.population)
        
        self.best_cost = torch.min(self.costs)
        self.best_individual = self.population[torch.argmin(self.costs)]
        
    def _init_population(self):
        population = []
        base = torch.arange(self.problem_size, device=self.device)
        for _ in range(self.pop_size):
            perm = base[torch.randperm(self.problem_size)]
            population.append(perm)
        return torch.stack(population)
    
    def _evaluate_population(self, population):
        # population: (pop_size, problem_size)
        u = population
        v = torch.roll(u, shifts=1, dims=1)
        costs = self.distances[u, v].sum(dim=1)
        fitness = 1.0 / costs
        return fitness, costs

    def _tournament_selection(self):
        indices = torch.randint(low=0, high=self.pop_size, size=(self.tournament_size,), device=self.device)
        selected_costs = self.costs[indices]
        winner_idx = indices[torch.argmin(selected_costs)]
        return self.population[winner_idx].clone()

    def _ordered_crossover(self, parent1, parent2):
        start = torch.randint(0, self.problem_size - 1, (1,))
        end = torch.randint(start+1, self.problem_size, (1,))
        start = start.item()
        end = end.item()
        
        child = torch.full((self.problem_size,), -1, device=self.device)
        child[start:end] = parent1[start:end]
        
        p2_vals = parent2[~torch.isin(parent2, child)]
        mask = (child == -1)
        child[mask] = p2_vals
        
        return child

    def _swap_mutation(self, individual):
        a, b = torch.randint(0, self.problem_size, (2,))
        temp = individual[a].item()
        individual[a] = individual[b]
        individual[b] = temp
        return individual

    def _breed_new_population(self):
        new_population = []
        
        sorted_idx = torch.argsort(self.costs)
        elites = self.population[sorted_idx[:self.elitism_count]]
        for e in elites:
            new_population.append(e.clone())
        
        for _ in range(self.pop_size - self.elitism_count):
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            if torch.rand(()) < self.crossover_rate:
                child = self._ordered_crossover(parent1, parent2)
            else:
                child = parent1.clone()
            
            if torch.rand(()) < self.mutation_rate:
                child = self._swap_mutation(child)
            
            new_population.append(child)
        
        new_population = torch.stack(new_population)
        return new_population

    def run(self, n_generations):
        for _ in range(n_generations):
            self.population = self._breed_new_population()
            self.fitness, self.costs = self._evaluate_population(self.population)
            current_best_cost, idx = torch.min(self.costs, dim=0)
            if current_best_cost < self.best_cost:
                self.best_cost = current_best_cost
                self.best_individual = self.population[idx]
        return self.best_cost


@torch.no_grad()
def infer_instance_ga(distances, t_ga_diff, pop_size=50, crossover_rate=0.8, mutation_rate=0.2, 
                      elitism_count=1, tournament_size=5, device='cpu'):
    # Ensure no inf edges:
    # Replace very large or inf distances with a large finite number
    # Adjust this threshold and replacement as needed.
    distances = distances.clone()
    inf_mask = distances >= 1e9  # or use torch.isinf if your data has actual inf
    if torch.any(inf_mask):
        # Find the largest finite distance
        finite_distances = distances[~inf_mask]
        if finite_distances.numel() > 0:
            max_finite = finite_distances.max()
        else:
            # If all are inf, fallback
            max_finite = 1.0
        # Replace infinities with some larger finite value
        distances[inf_mask] = max_finite * 10.0

    ga = GeneticAlgorithmTSP(
        distances=distances, 
        pop_size=pop_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_count=elitism_count,
        tournament_size=tournament_size,
        device=device
    )

    results = torch.zeros(size=(len(t_ga_diff),), device=device)
    for i, t in enumerate(t_ga_diff):
        best_cost = ga.run(t)
        results[i] = best_cost
    return results

@torch.no_grad()
def test_genetic(dataset, pop_size=50, crossover_rate=0.8, mutation_rate=0.2, elitism_count=1, 
                 tournament_size=5, t_ga=[10,20,30], device='cpu'):
    _t_ga = [0] + t_ga
    t_ga_diff = [_t_ga[i+1]-_t_ga[i] for i in range(len(_t_ga)-1)]
    sum_results = torch.zeros(size=(len(t_ga_diff),), device=device)
    start = time.time()
    res = []
    for pyg_data, distances in dataset:
        results = infer_instance_ga(distances, t_ga_diff, pop_size=pop_size, crossover_rate=crossover_rate, 
                                    mutation_rate=mutation_rate, elitism_count=elitism_count, 
                                    tournament_size=tournament_size, device=device)
        res.append(results)
        sum_results += results
    end = time.time()
    return res, sum_results / len(dataset), end - start