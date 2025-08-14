import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from torch_geometric.utils import degree
import torch

random.seed(123)
np.random.seed(123)


class Graph:
    ''' graph class '''
    def __init__(self, nodes, edges, children, parents, features):
        self.nodes = nodes # set()
        self.edges = edges # dict{(src,dst): weight, }
        self.children = children # dict{node: set(), }
        self.parents = parents # dict{node: set(), }
        self.features = features
        # transfer children and parents to dict{node: list, }
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])

        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        self._adj = None
        self._from_to_edges = None
        self._from_to_edges_weight = None

    def get_children(self, node):
        ''' outgoing nodes '''
        return self.children.get(node, [])

    def get_parents(self, node):
        ''' incoming nodes '''
        return self.parents.get(node, [])

    def get_prob(self, edge):
        return self.edges[edge]

    def get_adj(self):
        ''' return scipy sparse matrix '''
        if self._adj is None:
            self._adj = np.zeros((self.num_nodes, self.num_nodes))
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge] # may contain weight
            self._adj = csr_matrix(self._adj)
        return self._adj

    def from_to_edges(self):
        ''' return a list of edge of (src,dst) '''
        if self._from_to_edges is None:
            self._from_to_edges_weight = list(self.edges.items())
            self._from_to_edges = [p[0] for p in self._from_to_edges_weight]
        return self._from_to_edges

    def from_to_edges_weight(self):
        ''' return a list of edge of (src, dst) with edge weight '''
        if self._from_to_edges_weight is None:
            self.from_to_edges()
        return self._from_to_edges_weight


def read_graph(path, ind=0, directed=False):
    ''' method to load edge as node pair graph '''
    parents = {}
    children = {}
    edges = {}
    nodes = set()
    features = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue
            row = line.split()
            src = int(row[0]) - ind
            dst = int(row[1]) - ind
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst)
            parents.setdefault(dst, set()).add(src)
            edges[(src, dst)] = 0.0
            if not(directed):
                # regard as undirectional
                children.setdefault(dst, set()).add(src)
                parents.setdefault(src, set()).add(dst)
                edges[(dst, src)] = 0.0

    for src, dst in edges:
        edges[(src, dst)] = 1.0 / len(parents[dst])
    features = np.random.rand(len(nodes), 3)

    return Graph(nodes, edges, children, parents, features)

def computeMC(graph, S, R):
    sources = set(S)
    inf = 0
    for _ in range(R):
        source_set = sources.copy()
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                    if not(child in source_set) and random.random() <= graph.edges[(curr_node, child)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        inf += len(source_set)
        
    return inf / R

def workerMC(x):
    ''' for multiprocessing '''
    return computeMC(x[0], x[1], x[2])

def computeRR(graph, S, R, cache=None):
    covered = 0
    generate_RR = False
    if cache is not None:
        if len(cache) > 0:
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True

    for i in range(R):
        # generate one set
        source_set = {random.randint(0, graph.num_nodes - 1)}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        # compute covered(RR) / number(RR)
        for s in S:
            if s in source_set:
                covered += 1
                break
        if generate_RR:
            cache.append(source_set)
    return covered * 1.0 / R * graph.num_nodes

def computeSIS(graph, S, beta=0.3, gamma=0.1, num_steps=1):
    status = {node: 'S' for node in range(graph.num_nodes)}
    infected_set = set()
    for i in S:
        i = i.item()
        status[i] = 'I'
        infected_set.add(i)

    for step in range(num_steps):
        new_status = status.copy()
        for node in range(graph.num_nodes):
            if status[node] == 'I':
                neighbors = []
                edges = graph.edge_index.t().tolist()
                for t in edges:
                    if t[0] == node or t[1] == node:
                        # print(t)
                        if t[0] == node:
                            neighbors.append(t[1])
                        else:
                            neighbors.append(t[0])

                for neighbor in neighbors:
                    if status[neighbor] == 'S' and random.random() < beta:
                        new_status[neighbor] = 'I'
                        infected_set.add(neighbor)

                if random.random() < gamma:
                    new_status[node] = 'S'

        status = new_status

    return len(infected_set)

def computeIC(graph, S, steps=1):
    influenced = set(S)
    new_influenced = set(S)
    deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes)

    def get_neighbors(node, edge_index):
        src, dst = edge_index
        neighbors = dst[src == node]
        return neighbors

    for _ in range(steps):
        current_influenced = set()
        for node in new_influenced:
            neighbors = get_neighbors(node, graph.edge_index)
            for neighbor in neighbors:
                if neighbor not in influenced:
                    p = 0.8
                    if random.random() < p:
                        current_influenced.add(neighbor)
        new_influenced = current_influenced
        influenced.update(new_influenced)
        int_set = set()
        for item in influenced:
            if isinstance(item, torch.Tensor):
                int_item = int(item.item())
            else:
                int_item = int(item)

            int_set.add(int_item)

        if not new_influenced:
            break
    return len(int_set)

def computeLT(graph, S, steps=1):
    influenced = set(S)
    new_influenced = set(S)

    for _ in range(steps):
        current_influenced = set()
        for node in new_influenced:
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in influenced:
                    neighbor_influence = sum(1 for n in graph.neighbors(neighbor) if n in influenced) / graph.degree(neighbor)
                    threshold = graph.nodes[neighbor]['threshold']
                    if neighbor_influence >= threshold:
                        current_influenced.add(neighbor)
        new_influenced = current_influenced
        influenced.update(new_influenced)
        if not new_influenced:
            break
    return len(influenced)


def workerRR(x):
    ''' for multiprocessing '''
    return computeRR(x[0], x[1], x[2])

def computeRR_inc(graph, S, R, cache=None, l_c=None):
    covered = 0
    generate_RR = False
    if cache is not None:
        if len(cache) > 0:
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True

    for i in range(R):
        source_set = {random.randint(0, graph.num_nodes - 1)}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        for s in S:
            if s in source_set:
                covered += 1
                break
        if generate_RR:
            cache.append(source_set)
    return covered * 1.0 / R * graph.num_nodes


if __name__ == '__main__':
    # path of the graph file
    path = "../soc-dolphins.txt"
    # number of parallel processes
    num_process = 5
    # number of trials
    num_trial = 10000
    # load the graph
    graph = read_graph(path, ind=1, directed=False)
    print('Generating seed sets:')
    list_S = []
    for _ in range(10):
      list_S.append(random.sample(range(graph.num_nodes), k=random.randint(3, 10)))
      print(f'({str(list_S[-1])[1:-1]})')

    print('Cached single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    RR_cache = []
    for S in list_S:
      time_start = time.time()
      es_infs.append(computeRR(graph, S, num_trial, cache=RR_cache))
      times.append(time.time() - time_start)
    time_2 = time.time()

    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')

    print('No-cache single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
      time_start = time.time()
      es_infs.append(computeRR(graph, S, num_trial))
      times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')

    print('Multi-process MC:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
      time_start = time.time()
      with Pool(num_process) as p:
        es_infs.append(statistics.mean(p.map(workerMC, [[graph, S, num_trial // num_process] for _ in range(num_process)])))
      times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')
