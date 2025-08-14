import random
import pickle
import numpy as np
import torch
import networkx as nx
from scipy.sparse import csr_matrix
import io, gzip, zipfile
from urllib.request import urlopen, Request


def _fetch_bytes(src: str) -> bytes:
    if src.startswith("http://") or src.startswith("https://"):
        req = Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as resp:
            return resp.read()
    else:
        with open(src, "rb") as f:
            return f.read()

def _parse_pajek_net_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("%")]
    n = 0
    edges = []
    is_directed = False

    i = 0
    while i < len(lines):
        ln = lines[i]
        low = ln.lower()
        if low.startswith("*vertices"):
            parts = ln.split()
            # e.g., *Vertices 198
            for tok in parts[1:]:
                if tok.isdigit():
                    n = int(tok); break
            i += 1
            # 跳过 n 行顶点定义（有的文件用引号+标签等）
            skipped = 0
            while i < len(lines) and skipped < n:
                if lines[i].startswith("*"):
                    break
                skipped += 1
                i += 1
            continue

        if low.startswith("*edges") or low.startswith("*arcs"):
            is_directed = low.startswith("*arcs")
            i += 1
            while i < len(lines) and not lines[i].startswith("*"):
                parts = lines[i].split()
                if len(parts) >= 2:
                    try:
                        u = int(float(parts[0]))
                        v = int(float(parts[1]))
                        edges.append((u, v))
                    except ValueError:
                        pass
                i += 1
            continue

        i += 1

    if n <= 0 and edges:
        n = max(max(u, v) for u, v in edges)
    return n, edges, is_directed

def load_jazz_graph(
    src: str = "https://www-personal.umich.edu/~mejn/netdata/jazz.zip",
    *,
    largest_connected_component: bool = True,
    relabel_contiguous: bool = True,
) -> nx.Graph:

    raw = _fetch_bytes(src)

    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        text = io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding="utf-8", errors="ignore").read()
        n, edges_1b, is_directed = _parse_pajek_net_text(text)
    elif zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            net_names = [n for n in zf.namelist() if n.lower().endswith(".net")]
            if not net_names:
                raise ValueError("Zip 中未找到 .net 文件")
            with zf.open(net_names[0]) as f:
                text = f.read().decode("utf-8", errors="ignore")
        n, edges_1b, is_directed = _parse_pajek_net_text(text)
    else:

        text = raw.decode("utf-8", errors="ignore")
        n, edges_1b, is_directed = _parse_pajek_net_text(text)

    G = nx.Graph()
    for (u1, v1) in edges_1b:
        u, v = u1 - 1, v1 - 1
        if u == v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        if not G.has_edge(a, b):
            G.add_edge(a, b)

    if largest_connected_component and G.number_of_nodes() > 0:
        comps = list(nx.connected_components(G))
        giant = max(comps, key=len)
        G = G.subgraph(giant).copy()

    if relabel_contiguous:
        mapping = {old: i for i, old in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)

    return G

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


def process_jazz_dataset(src, model='IC'):
    G = load_jazz_graph(src)

    for v in G.nodes():
        G.nodes[v]['threshold'] = random.uniform(0.3, 0.6)

    num_nodes = G.number_of_nodes()
    k = int(h * num_nodes)
    random.seed(42)
    nodes_list = list(G.nodes())
    seeds_combinations = [random.sample(nodes_list, k) for _ in range(100)]

    model_function = {
        'IC': independent_cascade,
        'LT': linear_threshold,
        'ICI': independent_cascade_invitation_ici,
        'SIS': sis_model
    }[model]

    iterations = 1000
    inverse_pairs = np.zeros((100, num_nodes, 2), dtype=np.float32)
    for i, seeds in enumerate(seeds_combinations):
        node_prob = {node: 0 for node in G.nodes()}
        for _ in range(iterations):
            influenced = model_function(G, seeds)
            for node in influenced:
                node_prob[node] += 1
        for node in node_prob:
            node_prob[node] /= iterations
        for node in seeds:
            node_prob[node] = 1.0
        for j, node in enumerate(G.nodes()):
            inverse_pairs[i, j, 0] = 1 if node in seeds else 0
            inverse_pairs[i, j, 1] = node_prob[node]

    inverse_pairs = torch.tensor(inverse_pairs, dtype=torch.float32)

    adj_csr = nx.adjacency_matrix(G, dtype=np.float32)  # scipy.sparse.csr_matrix

    Graph = {'inverse_pairs': inverse_pairs, 'graph': G, 'adj': adj_csr}
    file_out = f'jazz-{model}-{int(h*100)}.pkl'
    with open(file_out, 'wb') as f:
        pickle.dump(Graph, f)
    print(f"[Jazz] Saved to {file_out} | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")


if __name__ == "__main__":
    bili = [0.05, 0.1, 0.2, 0.3, 0.4]
    jazz_src = "https://www-personal.umich.edu/~mejn/netdata/jazz.zip"
    for h in bili:
        for model in ['IC', 'ICI', 'LT', 'SIS']:
            process_jazz_dataset(jazz_src, model)