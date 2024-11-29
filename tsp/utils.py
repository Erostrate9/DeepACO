import torch
from torch_geometric.data import Data
from data_helper import load_symmetric_tsp
 
def gen_distance_matrix(tsp_coordinates):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances
    
def gen_pyg_data(tsp_coordinates, k_sparse):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    n_nodes = len(tsp_coordinates)
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances, 
                                           k=k_sparse, 
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)
    pyg_data = Data(x=tsp_coordinates, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

def gen_pyg_data_fully_connected(tsp_coordinates):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        pyg_data: Fully connected graph as a PyG Data instance (excluding self-loops)
        distances: Distance matrix
    '''
    n_nodes = len(tsp_coordinates)
    distances = gen_distance_matrix(tsp_coordinates)  # Generate the full distance matrix

    # Create edge indices for a fully connected graph (excluding self-loops)
    source_nodes, target_nodes = torch.where(torch.arange(n_nodes).view(-1, 1) != torch.arange(n_nodes))
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    # Assign edge attributes (distances)
    edge_attr = distances[source_nodes, target_nodes].view(-1, 1)

    # Return the fully connected graph as a PyG Data object
    pyg_data = Data(x=tsp_coordinates, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

def load_val_dataset(n_node, k_sparse, device):
    val_list = []
    val_tensor = torch.load(f'../data/tsp/valDataset-{n_node}.pt')
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse)
        val_list.append((data, distances))
    return val_list

def load_test_dataset(n_node, k_sparse, device):
    val_list = []
    val_tensor = torch.load(f'../data/tsp/testDataset-{n_node}.pt')
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse)
        val_list.append((data, distances))
    return val_list

def load_single_test_instance(file_path='../data/tsp/TSPLIB/berlin52.tsp', device='cpu'):
    val_list = []
    _, _, tsp_coordinates = load_symmetric_tsp(file_path)
    tsp_coordinates = torch.Tensor(tsp_coordinates).to(device)
    
    data, distances = gen_pyg_data_fully_connected(tsp_coordinates)
    val_list.append((data, distances))
    return val_list

if __name__ == "__main__":
    pass