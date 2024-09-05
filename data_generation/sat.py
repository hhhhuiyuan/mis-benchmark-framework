from data_generation.generator import DataGenerator
from pathlib import Path

from pysat.formula import CNF
import networkx as nx
import numpy as np
import pickle
from tqdm import tqdm

class SATGraphDataGenerator(DataGenerator):

    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def _build_graph(self, cnf_file, output_file, gen_labels, weighted, solver):
        cnf = CNF(cnf_file)
        nv = cnf.nv
        clauses = list(filter(lambda x: x, cnf.clauses))
        # pick the first 100 clauses
        # clauses = cnf.clauses[:100]
        
        ind = { k:[] for k in np.concatenate([np.arange(1, nv+1), -np.arange(1, nv+1)]) }
        edges = []

        for i, clause in enumerate(clauses):
            a = clause[0]
            b = clause[1]
            c = clause[2]
            aa = 3 * i + 0
            bb = 3 * i + 1
            cc = 3 * i + 2
            ind[a].append(aa)
            ind[b].append(bb)
            ind[c].append(cc)
            edges.append((aa, bb))
            edges.append((aa, cc))
            edges.append((bb, cc))

        for i in np.arange(1, nv+1):
            for u in ind[i]:
                for v in ind[-i]:
                    edges.append((u, v))

        G = nx.from_edgelist(edges)

        if weighted:
            weight_mapping = { vertex: weight for vertex, weight in zip(G.nodes, self.random_weight(G.number_of_nodes())) }
            nx.set_node_attributes(G, values = weight_mapping, name='weight')

        if gen_labels:
            if not weighted:
                mis_uw, _, _ = self._call_gurobi_solver(G) if solver == 'gurobi'\
                            else self._call_kamis_solver(G)
                label_mapping = { vertex: int(vertex in mis_uw) for vertex in G.nodes }
                nx.set_node_attributes(G, values = label_mapping, name='label')
            else:
                mis_w, _, obj = self._call_gurobi_solver(G, weighted=True) if solver == 'gurobi'\
                            else self._call_kamis_solver(G, weighted=True)
                label_mapping = { vertex: int(vertex in mis_w) for vertex in G.nodes }
                nx.set_node_attributes(G, values = label_mapping, name='label')
                G.graph['objective'] = obj

        # write graph object to output file
        with open(output_file, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    def generate(self, gen_labels = False,  weighted = False, label_solver = "kamis"):
        for f in tqdm(self.input_path.rglob("*.cnf")):
            self._build_graph(f, self.output_path / (f.stem + ".gpickle"), gen_labels, weighted, label_solver)