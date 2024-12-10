import time
import torch
import os

from net import Net
from aco import ACO
from utils import gen_pyg_data, load_val_dataset
from tqdm import tqdm

torch.manual_seed(1234)

lr = 3e-4
EPS = 1e-10
T=5
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def train_instance(model, optimizer, pyg_data, distances, n_ants):
    model.train()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    
    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat,
        distances=distances,
        device=device
        )
    
    costs, log_probs = aco.sample()
    baseline = costs.mean()
    reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()

def infer_instance(model, pyg_data, distances, n_ants):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat,
        distances=distances,
        device=device
        )
    costs, log_probs = aco.sample()
    aco.run(n_iterations=T)
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    best_aco_cost = aco.lowest_cost
    return baseline.item(), best_sample_cost.item(), best_aco_cost.item()

def train_epoch(n_node,
                n_ants, 
                k_sparse, 
                epoch, 
                steps_per_epoch, 
                net, 
                optimizer
                ):
    for _ in range(steps_per_epoch):
        instance = torch.rand(size=(n_node, 2), device=device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse)
        train_instance(net, optimizer, data, distances, n_ants)


@torch.no_grad()
def validation(n_ants, epoch, net, val_dataset, animator=None):
    sum_bl, sum_sample_best, sum_aco_best = 0, 0, 0
    
    for data, distances in val_dataset:
        bl, sample_best, aco_best = infer_instance(net, data, distances, n_ants)
        sum_bl += bl; sum_sample_best += sample_best; sum_aco_best += aco_best
    
    n_val = len(val_dataset)
    avg_bl, avg_sample_best, avg_aco_best = sum_bl/n_val, sum_sample_best/n_val, sum_aco_best/n_val
    if animator:
        animator.add(epoch+1, (avg_bl, avg_sample_best, avg_aco_best))
    return avg_bl, avg_sample_best, avg_aco_best

def train(n_node, n_ants, steps_per_epoch, epochs, k_sparse=None, savepath = "../pretrained/tsp"):
    k_sparse = k_sparse or n_node//10
    net = Net().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    val_list = load_val_dataset(n_node, k_sparse, device)
    
    avg_bl, avg_best, avg_aco_best = validation(n_ants, -1, net, val_list)
    val_results = [(avg_bl, avg_best, avg_aco_best)]
    
    sum_time = 0
    best_epoch = -1
    best_avg_aco_best = avg_aco_best
    for epoch in tqdm(range(0, epochs), f'Training DeepACO (without LC)...'):
        start = time.time()
        train_epoch(n_node, n_ants, k_sparse, epoch, steps_per_epoch, net, optimizer)
        sum_time += time.time() - start
        avg_bl, avg_sample_best, avg_aco_best = validation(n_ants, epoch, net, val_list)
        val_results.append((avg_bl, avg_sample_best, avg_aco_best))
        torch.save(net.state_dict(), os.path.join(savepath, f'tsp{n_node}-last.pt'))
        if avg_aco_best <= best_avg_aco_best:
            torch.save(net.state_dict(), os.path.join(savepath, f'tsp{n_node}-best.pt'))
            best_avg_aco_best = avg_aco_best
            best_epoch = epoch
    for epoch in range(-1, epochs):
        print(f'epoch {epoch}:', val_results[epoch+1])    
    print('total training duration:', sum_time)
    print('Final total training epochs:', epochs)
    print(f'Best parameters was obtained from epoch {best_epoch}')
    print('Final total training instances processed:', epochs*steps_per_epoch)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar='N', type=int, help="Problem scale")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str, 
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-a", "--ants", type=int, default=20, help="Number of ants (in ACO algorithm)")
    parser.add_argument("-s", "--steps", type=int, default=128, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Epochs to run")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/tsp",
                        help="The directory to store checkpoints")
    opt = parser.parse_args()
    
    if os.path.isdir(opt.output) is False:
        os.mkdir(opt.output)        

    lr = opt.lr
    device = opt.device
    n_node = opt.nodes
    
    print(f'The device to train NNs: {device}')

    K = {20:10, 50:20, 100:20, 50:500, 1000:100}
    k_sparse = K[n_node] if n_node in K else n_node//10
    print(f'k_sparse: {k_sparse}')

    train(
        opt.nodes, 
        opt.ants, 
        opt.steps, 
        opt.epochs,
        k_sparse = k_sparse,
        savepath = opt.output,
    )