import os
import torch
from torch_geometric.data import Data
from data_helper import load_symmetric_tsp
import numpy as np

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

def gen_pyg_data_fully_connected(tsp_coordinates, start_node = None):
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
    if start_node is None:
        node_feature = tsp_coordinates
    else:
        # node_feature = torch.hstack([tsp_coordinates, torch.zeros((n_nodes,1), device=tsp_coordinates.device)])
        # node_feature[start_node, 2] = 1.0
        node_feature = torch.zeros((n_nodes,1), device=tsp_coordinates.device, dtype=tsp_coordinates.dtype)
        node_feature[start_node, 0] = 1.0
    # Return the fully connected graph as a PyG Data object
    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

def gen_pyg_data(tsp_coordinates, k_sparse, start_node = None):
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

    if start_node is None:
        node_feature = tsp_coordinates
    else:
        # node_feature = torch.hstack([tsp_coordinates, torch.zeros((n_nodes,1), device=tsp_coordinates.device)])
        # node_feature[start_node, 2] = 1.0
        node_feature = torch.zeros((n_nodes,1), device=tsp_coordinates.device, dtype=tsp_coordinates.dtype)
        node_feature[start_node, 0] = 1.0
    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

def load_val_dataset(n_node, k_sparse, device, start_node = None):
    if not os.path.isfile(f'../data/tsp/valDataset-{n_node}.pt'):
        val_tensor = torch.rand((50, n_node, 2))
        torch.save(val_tensor, f'../data/tsp/valDataset-{n_node}.pt')
    else:
        val_tensor = torch.load(f'../data/tsp/valDataset-{n_node}.pt')

    val_list = []
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse, start_node = start_node)
        val_list.append((data, distances))
    return val_list

def load_test_dataset(n_node, k_sparse, device, start_node = None, filename = None):
    val_list = []
    filename = filename or f'../data/tsp/testDataset-{n_node}.pt'
    val_tensor = torch.load(filename)
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse, start_node = start_node)
        val_list.append((data, distances))
    return val_list

def load_TSPLIB_test_instance(dir='../data/tsp/TSPLIB/', file_names=['berlin52.tsp'], device='cpu', start_node = None, k_sparse=None, normalized=False):
    assert os.path.isdir(dir)
    val_list = []
    if not file_names:
        file_names = [f for f in os.listdir(dir) if f.endswith(".tsp")]
    for file_name in file_names:
        file_path = os.path.join(dir, file_name)
        if not os.path.exists(file_path):
            continue
        _, _, tsp_coordinates = load_symmetric_tsp(file_path)
        if normalized:
            min_val = np.min(tsp_coordinates)
            max_val = np.max(tsp_coordinates)
            if min_val == max_val:
                tsp_coordinates = np.zeros_like(tsp_coordinates)  # Assign all values as 0
            else:
                tsp_coordinates = (tsp_coordinates - min_val) / (max_val - min_val)
        tsp_coordinates = torch.Tensor(tsp_coordinates).to(device)
        if k_sparse==None:
            data, distances = gen_pyg_data_fully_connected(tsp_coordinates)
        else:
            data, distances = gen_pyg_data(tsp_coordinates, k_sparse=k_sparse, start_node = start_node)
        val_list.append((data, distances))
    return val_list

if __name__ == "__main__":
    pass