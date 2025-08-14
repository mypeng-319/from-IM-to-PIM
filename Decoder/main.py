import numpy as np
import torch
import contextlib

import argparse
import pickle
import random
import scipy.sparse as sp
from torch_geometric.data import Data

from Decoder.dt.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from Decoder.dt.models.decision_transformer1 import DecisionTransformer
from Decoder.dt.models.mlp_bc import MLPBCModel
from Decoder.dt.training.act_trainer import ActTrainer
from Decoder.dt.training.seq_trainer import SequenceTrainer
from Decoder.dt.evaluation import environment
from torch_geometric.utils import to_networkx, from_networkx


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
        dataset,
        method,
        budget,
):
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], dataset
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    def read_fb_food_data(file_path):
        edges = []
        with open(file_path, 'r') as file:
            for line in file:
                node1, node2 = map(int, line.strip().split())
                edges.append((node1, node2))
        return edges

    def normalize_adj(mx):
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    if dataset=='cora':
        with open('data/cora/cora.SG', 'rb') as f:
            graph = pickle.load(f)
    elif dataset=='citeseer':
        with open('data/citeseer/citeseer.SG', 'rb') as f:
            graph = pickle.load(f)
    elif dataset=='fb_food':
        with open('data/fb_food/fb_food.SG', 'rb') as f:
            graph = pickle.load(f)
    elif dataset=='fb_tvshow':
        with open('data/fb_tvshow/fb_tvshow.SG', 'rb') as f:
            graph = pickle.load(f)
    elif dataset=='friendster':
        with open('data/friendster/com-friendster.ungraph.txt.gz', 'rb') as f:
            graph = pickle.load(f)
    elif dataset=='jazz':
        with open('data/jazz/jazz.SG', 'rb') as f:
            graph = pickle.load(f)
    elif dataset=='network':
        with open('data/network/network.SG', 'rb') as f:
            graph = pickle.load(f)
    elif dataset=='wiki':
        with open('data/wiki/wiki2.SG', 'rb') as f:
            graph = pickle.load(f)
    elif dataset=='youtube':
        with open('data/youtube/com-youtube.ungraph.txt.gz', 'rb') as f:
            graph = pickle.load(f)

    adj = graph['adj']
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.Tensor(adj.toarray()).to_sparse()

    edge_index = adj.indices()
    edge_attr = adj.values()

    num_nodes = adj.shape[0]

    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    G = to_networkx(data, to_undirected=True)
    G = from_networkx(G)
    num = G.num_nodes
    m = method
    budget = float(budget)

    if env_name == 'IM':
        data_train = []
        data_train.append(G)
        env = environment.Environment(env_name, data_train, budget=int(budget*num), method=m, use_cache=True)
        max_ep_len = int(budget*num)
        env_targets = [2000000]
        scale = 1000.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]

    state_dim = G.num_nodes
    act_dim = G.num_nodes

    if dataset=='cora':
        dataset_path = f'data/cora/cora-{m}-{int(budget*100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset=='citeseer':
        dataset_path = f'data/citeseer/citeseer-{m}-{int(budget * 100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset == 'fb_food':
        dataset_path = f'data/fb_food/fb_food-{m}-{int(budget * 100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset == 'fb_tvshow':
        dataset_path = f'data/fb_tvshow/fb_tvshow-{m}-{int(budget * 100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset == 'jazz':
        dataset_path = f'data/jazz/jazz-{m}-{int(budget * 100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset == 'network':
        dataset_path = f'data/network/network-{m}-{int(budget * 100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset == 'wiki':
        dataset_path = f'data/wiki/wiki2-{m}-{int(budget * 100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset == 'youtube':
        dataset_path = f'data/youtube/youtube-{m}-{int(budget * 100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    elif dataset == 'friendster':
        dataset_path = f'data/friendster/friendster-{m}-{int(budget * 100)}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['states'])
        traj_lens.append(len(path['states']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset} {m} {budget}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['states'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminal' in traj:
                d.append(traj['terminal'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length, R = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
                max_return = max(returns)
                influ1 = float(max_return/state_dim)
                influ2 = float(R / state_dim)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'dataset_{dataset}_diffusion_model_{m}_budget_{budget}_influence_spread_REWARD': influ1,
                f'dataset_{dataset}_diffusion_model_{m}_budget_{budget}_nfluence_spread_R': influ2,
            }

        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )


    with open(f'{dataset}_{m}_{budget}_output.txt', 'a') as f:
        with contextlib.redirect_stdout(f):
            for iter in range(variant['max_iters']):
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='IM')
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=6)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    for dataset in ['cora','citeseer','fb_food','fb_tvshow','friendster','jazz','wiki','youtube']:
        for method in ['LT','IC','SIS','ICI']:
            for budget in [0.05, 0.1, 0.2, 0.3, 0.4]:
                experiment(f'{dataset}_PIM', variant=vars(args), dataset=dataset, method=method, budget=budget)