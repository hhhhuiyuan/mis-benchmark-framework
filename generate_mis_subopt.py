#run by $python data/mis-benchmark-framework/generate_mis_subopt.py
import numpy as np
import pickle
import glob
import networkx as nx
import os
from tqdm import tqdm

def findMIS(AD, priority, weight):
    # Check if the adjacency matrix is square
    N, M = AD.shape
    if N != M:
        raise ValueError('Adjacency matrix AD must be square')

    x = -np.ones(N)
    nID = np.arange(N)

    # Find the vertices with the minimum degree
    degree = np.sum(AD * weight, axis=1)
    md = np.min(degree)
    minDeg = degree == md  # vertices with the minimum degree

    if np.sum(minDeg) > 1:  # If more than one vertex has the min number of adjacent vertices
        support = np.zeros_like(minDeg, dtype=int)
        # For each minimum degree vertex, consider the support (sum of degrees of all its adjacent vertices)
        for i in np.where(minDeg)[0]:
            support[i] = np.sum(degree[AD[i, :] > 0])
        # Find the vertices with the maximum support
        ms = np.max(support)
        if ms > 0:
            # minDeg_maxSup -> vertices with minimum degree which maximize the support
            minDeg_maxSup = np.where(support == ms)[0]
        else:  # if support is full of 0 -> all the vertices minDeg have deg=0
            minDeg_maxSup = np.where(minDeg)[0]
    else:
        minDeg_maxSup = np.where(minDeg)[0]

    if len(minDeg_maxSup) > 1:
        j = np.argmin(priority[minDeg_maxSup])
        nodSel = minDeg_maxSup[j]
    else:
        nodSel = minDeg_maxSup[0]

    x[nodSel] = 1
    x[AD[nodSel, :] > 0] = 0
    assigned = x > -1
    AD = AD[~assigned, :][:, ~assigned]
    nID = nID[~assigned]
    priority = priority[~assigned]
    weight = weight[~assigned]

    if AD.size > 0:
        x[nID] = findMIS(AD, priority, weight)

    return x


mis_folder = '../data/shared/huiyuan/mis100/weighted_ER_train_1234/*gpickle'
new_folder = '../data/shared/huiyuan/mis100/weighted_ER_subbopt_train_1234/' 
os.makedirs(new_folder, exist_ok=True)

mis_files = glob.glob(mis_folder)
gap = []
opt = []

for mis_file in tqdm(mis_files):
    with open(mis_file , "rb") as f:
        graph = pickle.load(f)
    
    # nonopt_flag = 'non-optimal' in mis_file
    weight = nx.get_node_attributes(graph, 'weight')
    num_nodes = graph.number_of_nodes()
    
    if not weight:
        weight = np.ones(num_nodes)
    else:
        weight = np.array(list(weight.values()))
       
    # if nonopt_flag:
    #     node_labels = [_[1] for _ in graph.nodes(data='nonoptimal_label')]
    # else:
    #     node_labels = [_[1] for _ in graph.nodes(data='label')]
    opt_obj = graph.graph["objective"]
    
    adj = np.array(nx.adjacency_matrix(graph).todense())
    priority = np.array(list(range(num_nodes))) + 1
    
    result = findMIS(adj, priority, weight)
    subopt_obj = np.sum(result * weight)
    
    gap.append(opt_obj - subopt_obj)
    opt.append(opt_obj)
    
    graph.graph["objective"]= subopt_obj
    subopt_label = {i:result[i] for i in range(num_nodes)}   
    nx.set_node_attributes(graph, subopt_label, 'label')
    
    new_filename = f"subopt_{os.path.basename(mis_file)}"
    nx.write_gpickle(graph, os.path.join(new_folder, new_filename))
    
print(f"Average gap: {np.mean(gap)}, Average opt: {np.mean(opt)}")
    
    