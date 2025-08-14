import random
import pickle
import numpy as np
import torch
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid


#IC
def independent_cascade(G, seeds, steps=100):
    influenced = set(seeds)
    new_influenced = set(seeds)

    for _ in range(steps):
        current_influenced = set()
        for node in new_influenced:
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in influenced:
                    p = 1 / G.degree(neighbor)
                    if random.random() < p:
                        current_influenced.add(neighbor)
        new_influenced = current_influenced
        influenced.update(new_influenced)
        if not new_influenced:
            break
    return influenced

#ICI
def independent_cascade_invitation_ici(
    G,
    seeds,
    beta=0.5,
    gamma=0.5,
    steps=100,
    edge_prob="auto",
    rng=None,
):
    if rng is None:
        rng = random

    def p_uv(u, v):
        if callable(edge_prob):
            return float(edge_prob(u, v, G))
        if isinstance(edge_prob, (int, float)):
            return float(edge_prob)

        if edge_prob == "auto":
            deg = G.degree(v)
            return 1.0 / deg if deg > 0 else 0.0

        data = G.get_edge_data(u, v, default={})

        if isinstance(data, dict) and "weight" in data and edge_prob == "weight":
            return float(data["weight"])
        if isinstance(data, dict):
            val = data.get(edge_prob, None)
            if val is not None:
                return float(val)

        deg = G.degree(v)
        return 1.0 / deg if deg > 0 else 0.0

    seeds = set(seeds)

    invited = set(seeds)       # V_e
    accepted = set(seeds)      # V_a
    inviters = set(seeds)      # V_r（frontier）

    for _ in range(steps):
        if not inviters:
            break

        new_invitees = set()
        for u in inviters:
            for v in G.neighbors(u):
                if v not in invited:
                    if rng.random() < p_uv(u, v):
                        new_invitees.add(v)
                        invited.add(v)

        if not new_invitees:
            inviters = set()
            break

        new_acceptors = set()
        for v in new_invitees:
            if rng.random() < beta:
                accepted.add(v)
                new_acceptors.add(v)

        next_inviters = set()
        for v in new_acceptors:
            if rng.random() < gamma:
                next_inviters.add(v)

        inviters = next_inviters

    return accepted


#LT
def linear_threshold(G, seeds, steps=100):
    influenced = set(seeds)
    new_influenced = set(seeds)

    for _ in range(steps):
        current_influenced = set()
        for node in new_influenced:
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in influenced:
                    neighbor_influence = sum(1 for n in G.neighbors(neighbor) if n in influenced) / G.degree(neighbor)
                    threshold = G.nodes[neighbor]['threshold']
                    if neighbor_influence >= threshold:
                        current_influenced.add(neighbor)
        new_influenced = current_influenced
        influenced.update(new_influenced)
        if not new_influenced:
            break
    return influenced


#SIS
def sis_model(G, seeds, beta=0.001, mu=0.001, steps=100):
    infected = set(seeds)
    all_nodes = set(G.nodes)

    for _ in range(steps):
        new_infected = set()
        for node in infected:
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in infected and random.random() < beta:
                    new_infected.add(neighbor)
            if random.random() >= mu:
                new_infected.add(node)
        infected = new_infected
    return infected


def process_dataset(dataset_name, model='IC'):
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    data = dataset[0]

    G = to_networkx(data, to_undirected=True)

    for v in G.nodes():
        G.nodes[v]['threshold'] = random.uniform(0.3, 0.6)

    num_nodes = data.num_nodes
    k = int(h * num_nodes)

    random.seed(42)
    seeds_combinations = [random.sample(list(G.nodes), k) for _ in range(100)]

    inverse_pairs = np.zeros((100, num_nodes, 2))

    model_function = {
        'IC': independent_cascade,
        'LT': linear_threshold,
        'SIS': sis_model,
        'ICI': independent_cascade_invitation_ici
    }[model]

    iterations = 1000
    for i, seeds in enumerate(seeds_combinations):
        node_probabilities = {node: 0 for node in G.nodes}

        for _ in range(iterations):
            influenced = model_function(G, seeds)
            for node in influenced:
                node_probabilities[node] += 1

        node_probabilities = {node: prob / iterations for node, prob in node_probabilities.items()}

        for node in seeds:
            node_probabilities[node] = 1.0

        for j, node in enumerate(G.nodes):
            initial_prob = 1 if node in seeds else 0
            final_prob = node_probabilities[node]
            inverse_pairs[i, j, 0] = initial_prob
            inverse_pairs[i, j, 1] = final_prob

    inverse_pairs = torch.tensor(inverse_pairs, dtype=torch.float32)

    pyg_graph = from_networkx(G)

    adj = nx.adjacency_matrix(G).todense()
    adj = np.array(adj)

    adj_csr = csr_matrix(adj)

    Graph = {'inverse_pairs': inverse_pairs, 'graph': pyg_graph, 'adj': adj_csr}

    file_path = f'{dataset_name}-{model}-{int(h*100)}.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(Graph, file)

    print(f"Graph dictionary saved to {file_path}")

bili = [0.05, 0.1, 0.2, 0.3, 0.4]
for h in bili:
    for dataset_name in ['cora', 'citeseer']:
        for model in ['IC', 'LT', 'ICI', 'SIS']:
            process_dataset(dataset_name, model)