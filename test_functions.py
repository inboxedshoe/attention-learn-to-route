from problems.vrp.problem_vrp import VRPDataset
from problems.vrp.problem_vrp import CVRP
import time 
import torch

data = VRPDataset( size=10)
problem = CVRP()

start = time.time()

distance_matrix = torch.cdist(data[1]["loc"], data[1]["loc"], p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
neighbors = torch.topk(distance_matrix, 5, largest = False)

print(time.time()-start)

print(neighbors.indices)

