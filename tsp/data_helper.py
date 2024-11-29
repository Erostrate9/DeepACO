import os
import subprocess
import shutil
import numpy as np

def clone_tsplib():
    """Step 1: inspect if dir tsplib exists. Fetch the dataset if it doesn't exist"""
    if not os.path.exists('tsplib') or not os.listdir('tsplib'):
        print("tsplib is empty or doesn't exist, trying to fetch tsplib dataset.")
        subprocess.run(['git', 'clone', 'https://github.com/mastqe/tsplib.git'])
    else:
        print("tsplib exists")

def create_target_dir():
    """Step 2: create dir tsplib/TSP_EUC_2D"""
    target_dir = os.path.join('tsplib', 'TSP_EUC_2D')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"{target_dir} is created")
    else:
        print(f"{target_dir} exists")

def filter_and_copy_files():
    """Step 3: find all TSP EUC_2D .tsp files"""
    tsplib_dir = 'tsplib'
    target_dir = os.path.join(tsplib_dir, 'TSP_EUC_2D')
    file_count = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
    if file_count >= 78:
        return
    for file_name in os.listdir(tsplib_dir):
        if file_name.endswith('.tsp'):
            file_path = os.path.join(tsplib_dir, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            type_found = False
            edge_weight_type_found = False
            tsp_type = None
            edge_weight_type = None

            for line in lines[:10]:
                key = line.split(':')[0].strip()
                value = line.split(':')[1].strip()
                if 'TYPE' == key:
                    tsp_type = value
                    type_found = True
                if 'EDGE_WEIGHT_TYPE' == key:
                    edge_weight_type = value
                    edge_weight_type_found = True
                if type_found and edge_weight_type_found:
                    break
            if tsp_type == 'TSP' and edge_weight_type == 'EUC_2D':
                # print(f"{file_name} is TSP and EUC_2D")
                shutil.copy(file_path, target_dir)

def fetch_dataset():
    clone_tsplib()
    create_target_dir()
    filter_and_copy_files()


def get_dis_matrix_from_coords(coords):
    """Given list of coordinates, return a matrix denotes 2D Euclidean distance between every pair of nodes.
    The Euclidean distance between coord1 = (x1,y1) and coord2 = (x2,y2) is computed as math.sqrt(sum([(coord1[i]-coord2[i])**2 for i in range(len(coord))])])

    Args:
        coords: List[List[float]], each element coord is (x, y) where x and y are both float number.
    """
    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    float_dis_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    int_dis_matrix = np.rint(float_dis_matrix).astype(int)
    res = int_dis_matrix.tolist()
    for i in range(len(res)):
        res[i][i]=float('inf')
    return res

# def get_dis_matrix_from_coords(coords):
#     """Given list of coordinates, return a matrix denotes 2D Euclidean distance between every pair of nodes.
#         coords: List[List[float]], each element coord is (x, y) where x and y are both float number.
#         The Euclidean distance between coord1 = (x1,y1) and coord2 = (x2,y2) is computed as math.sqrt(sum([(coord1[i]-coord2[i])**2 for i in range(len(coord))])])
#     """
#     def euc_dis(a, b):
#         assert len(a)==len(b)
#         d = len(a)
#         float_dis = math.sqrt(sum([(a[i]-b[i])**2 for i in range(d)]))
#         return int(float_dis + 0.5)
#     # dimension of TSP, i.e., # of nodes
#     n = len(coords)
#     matrix = [[float('inf')]*n for _ in range(n)]
#     for i in range(n):
#         for j in range(n):
#             if i==j:
#                 continue
#             matrix[i][j] = euc_dis(coords[i], coords[j])
#     return matrix


def load_symmetric_tsp(file_path):
    """load symmetric TSP matrix from a tsp file
    
    Args:
        file_path: str, the tsp file contains data of a symmetric TSP problem.
    Returns:
            matrix: List[List[int]], A 2-D list with size of n*n, where matrix[i][j] represents the distance between node i and node j
            fixed_edges: List[List[int]], 
            coords: coordinates
    """
    assert file_path.endswith('.tsp')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    type_found = False
    edge_weight_type_found = False
    tsp_type = None
    edge_weight_type = None
    fixed_edges_found = False
    fixed_edges = set()
    node_coord_found = False
    coords = None
    dimension = 0
    for line in lines:
        # read each line as <int> <float> <float>, the first <int> is the node index in matrix, other two floats is 2-D coordinate x_i and y_i
        if node_coord_found:
            if not line.strip() or line.strip()=='EOF':
                break
            params = line.strip().split()
            idx = int(params[0])-1
            coord = [float(x) for x in params[1:]]
            coords[idx] = coord
            continue
        # read each line as <int> <int>, the first <int> is the node index in matrix, other two floats is 2-D coordinate x_i and y_i
        if fixed_edges_found:
            if line.strip()=='-1':
                fixed_edges_found=False
                continue
            params = line.strip().split(' ')
            edge = tuple(sorted(int(x)-1 for x in params))
            fixed_edges.add(edge)
            continue

        if line.strip() == 'FIXED_EDGES_SECTION':
            fixed_edges_found = True
            continue
            
        if line.strip() == 'NODE_COORD_SECTION':
            node_coord_found = True
            continue
        key = line.split(':')[0].strip()
        value = line.split(':')[1].strip()
        if 'DIMENSION' == key:
            dimension = int(value)
            coords = [[0, 0] for _ in range(dimension)]
        if 'TYPE' == key:
            tsp_type = value
            type_found = True
        if 'EDGE_WEIGHT_TYPE' == key:
            edge_weight_type = value
            edge_weight_type_found = True
        if type_found and edge_weight_type_found:
            assert tsp_type == 'TSP' and edge_weight_type == 'EUC_2D'
    matrix = get_dis_matrix_from_coords(coords)
    return matrix, list(fixed_edges), coords