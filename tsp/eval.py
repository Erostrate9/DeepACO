import time
import torch
# from d2l.torch import Animator
from aco import ACO
from tqdm import tqdm

torch.manual_seed(12345)

EPS = 1e-10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse=None):
    if model:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    
        aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat,
        distances=distances,
        device=device
        )
    
    else:
        aco = ACO(
        n_ants=n_ants,
        distances=distances,
        device=device
        )
        if k_sparse:
            aco.sparsify(k_sparse)
        
    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    for i, t in enumerate(t_aco_diff):
        best_cost = aco.run(t)
        results[i] = best_cost
    return results

@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse=None):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    start = time.time()
    for pyg_data, distances in tqdm(dataset, desc="Testing Deep ACO Algorithm"):
    # for pyg_data, distances in dataset:
        results = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start
