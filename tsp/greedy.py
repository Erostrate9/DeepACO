import torch
import time

class GreedyTSP:
    def __init__(self, distances, device='cpu'):
        """
        Initialize the greedy TSP solver.

        Args:
            distances: torch tensor of shape (n, n) representing the distance matrix
            device: computational device (cpu or cuda)
        """
        self.distances = distances
        self.problem_size = distances.shape[0]
        self.device = device

    @torch.no_grad()
    def run(self):
        """
        Solve the TSP using the greedy algorithm.

        Returns:
            path: torch tensor of shape (problem_size,) representing the TSP tour
            total_cost: scalar value representing the cost of the tour
        """
        n = self.problem_size
        visited = torch.zeros(n, dtype=torch.bool, device=self.device)
        current_node = 0
        path = [current_node]
        visited[current_node] = True
        total_cost = 0.0

        for _ in range(n - 1):
            # Find the nearest unvisited neighbor
            unvisited_mask = ~visited
            distances_from_current = self.distances[current_node]
            distances_from_current[~unvisited_mask] = float('inf')  # Ignore visited nodes
            next_node = torch.argmin(distances_from_current).item()
            
            # Update path and cost
            total_cost += self.distances[current_node, next_node].item()
            path.append(next_node)
            visited[next_node] = True
            current_node = next_node

        # Complete the tour by returning to the starting node
        total_cost += self.distances[current_node, path[0]].item()
        path.append(path[0])

        return torch.tensor(path, device=self.device), total_cost

    @torch.no_grad()
    def infer_instance(self, distances):
        """
        Solve a single TSP instance using the greedy algorithm.

        Args:
            distances: torch tensor of shape (n, n) representing the distance matrix

        Returns:
            total_cost: scalar value representing the cost of the tour
        """
        n = distances.shape[0]
        visited = torch.zeros(n, dtype=torch.bool, device=self.device)
        current_node = 0
        visited[current_node] = True
        total_cost = 0.0

        for _ in range(n - 1):
            unvisited_mask = ~visited
            distances_from_current = distances[current_node]
            distances_from_current[~unvisited_mask] = float('inf')
            next_node = torch.argmin(distances_from_current).item()

            total_cost += distances[current_node, next_node].item()
            visited[next_node] = True
            current_node = next_node

        # Complete the tour by returning to the starting node
        total_cost += distances[current_node, 0].item()
        return total_cost

@torch.no_grad()
def test_greedy(dataset, greedy_solver):
    """
    Test the greedy algorithm on the given dataset.

    Args:
        dataset: List of test instances, where each instance is a tuple of (pyg_data, distances)
        greedy_solver: Instance of GreedyTSP class

    Returns:
        avg_cost: Average cost of the tours on the test dataset
        duration: Time taken to evaluate the dataset
    """
    total_cost = 0.0
    start = time.time()
    for _, distances in dataset:
        cost = greedy_solver.infer_instance(distances)
        total_cost += cost
    end = time.time()

    avg_cost = total_cost / len(dataset)
    return avg_cost, end - start

if __name__ == '__main__':
    torch.set_printoptions(precision=3, sci_mode=False)
    input = torch.rand(size=(5, 2))
    distances = torch.norm(input[:, None] - input, dim=2, p=2)
    distances[torch.arange(len(distances)), torch.arange(len(distances))] = float('inf')
    
    greedy_tsp = GreedyTSP(distances)
    path, cost = greedy_tsp.run()
    print("Path:", path)
    print("Cost:", cost)