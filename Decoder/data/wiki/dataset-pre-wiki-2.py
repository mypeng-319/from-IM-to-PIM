import random
import pickle
import numpy as np
import torch
import networkx as nx
from scipy.sparse import csr_matrix
import gzip, io  # NEW: for loader


def _open_text_auto(path: str, encoding: str = "utf-8"):

    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding=encoding)
    return open(path, "r", encoding=encoding)

def _iter_edges_snap_ungraph(path: str):
    with _open_text_auto(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            a = s.split()
            if len(a) < 2:
                continue
            u, v = int(a[0]), int(a[1])
            yield (u, v)

def load_snap_friendster(
    path: str,
    *,
    directed: bool = False,
    deduplicate: bool = True,
    remove_self_loops: bool = True,
    largest_connected_component: bool = True,
    relabel_contiguous: bool = True,
    edge_limit: int | None = None,
):
    G = nx.DiGraph() if directed else nx.Graph()
    seen = set() if (deduplicate and not directed) else None

    n_loaded = 0
    for u, v in _iter_edges_snap_ungraph(path):
        if remove_self_loops and u == v:
            continue
        if seen is not None:
            a, b = (u, v) if u <= v else (v, u)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            u, v = a, b
        G.add_edge(u, v)
        n_loaded += 1
        if edge_limit is not None and n_loaded >= edge_limit:
            break

    if largest_connected_component:
        comps = list(nx.connected_components(G)) if not directed else list(nx.weakly_connected_components(G))
        if comps:
            giant = max(comps, key=len)
            G = G.subgraph(giant).copy()

    if relabel_contiguous:
        mapping = {old: i for i, old in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)

    return G

def get_friendster_graph(
    path: str = "wiki-Vote.txt.gz",
    *,
    sample_edges: int | None = None,
):
    return load_snap_friendster(
        path=path,
        directed=False,
        deduplicate=True,
        remove_self_loops=True,
        largest_connected_component=True,
        relabel_contiguous=True,
        edge_limit=sample_edges,
    )


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
    invited = set(seeds)
    accepted = set(seeds)
    inviters = set(seeds)
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
def sis_model(G, seeds, beta=0.3, mu=0.001, steps=100):
    infected = set(seeds)
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

def process_friendster_dataset(file_path, model='IC'):
    G = get_friendster_graph(path=file_path)
    for v in G.nodes():
        G.nodes[v]['threshold'] = random.uniform(0.3, 0.6)
    num_nodes = G.number_of_nodes()
    k = int(h * num_nodes)
    random.seed(42)
    nodes_list = list(G.nodes())
    seeds_combinations = [random.sample(nodes_list, k) for _ in range(100)]
    inverse_pairs = np.zeros((100, num_nodes, 2), dtype=np.float32)
    model_function = {
        'IC': independent_cascade,
        'LT': linear_threshold,
        'ICI': independent_cascade_invitation_ici,
        'SIS': sis_model
    }[model]
    iterations = 1000
    for i, seeds in enumerate(seeds_combinations):
        node_probabilities = {node: 0 for node in G.nodes()}
        for _ in range(iterations):
            influenced = model_function(G, seeds)
            for node in influenced:
                node_probabilities[node] += 1
        node_probabilities = {node: prob / iterations for node, prob in node_probabilities.items()}
        for node in seeds:
            node_probabilities[node] = 1.0
        for j, node in enumerate(G.nodes()):
            initial_prob = 1 if node in seeds else 0
            final_prob = node_probabilities[node]
            inverse_pairs[i, j, 0] = initial_prob
            inverse_pairs[i, j, 1] = final_prob
    inverse_pairs = torch.tensor(inverse_pairs, dtype=torch.float32)
    adj_csr = nx.adjacency_matrix(G, dtype=np.float32)
    Graph = {'inverse_pairs': inverse_pairs, 'graph': G, 'adj': adj_csr}
    file_out = f'wiki2-{model}-{int(h*100)}.pkl'
    with open(file_out, 'wb') as f:
        pickle.dump(Graph, f)
    print(f"Graph dictionary saved to {file_out}  |  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")


bili = [0.05, 0.1, 0.2, 0.3, 0.4]
for h in bili:
    file_path = 'wiki-Vote.txt.gz'
    for model in ['IC', 'ICI', 'LT', 'SIS']:
        process_friendster_dataset(file_path, model)