import numpy as np
import torch
import Decoder.dt.evaluation.models as models
import random
from Encoder.graph import MyGT_gen
from torch_geometric.data import Data

def select_action(graph, state, epsilon, training=True, budget=None, device='cuda'):
        if not(training):
            graph_input = MyGT_gen(graph)
            with torch.no_grad():
                q_a = MyGT_gen(graph_input)
            q_a[state.nonzero()] = -1e5

            if budget is None:
                return torch.argmax(q_a).detach().clone()
            else:
                return torch.topk(q_a.squeeze(dim=1), budget)[1].detach().clone()
        # training
        available = (state == 0).nonzero()
        if epsilon > random.random():
            return random.choice(available)
        else:
            graph_input_h, graph_input_e = MyGT_gen(graph)
            graph_input = Data(x=graph_input_h, edge_index=graph_input_e)

            with torch.no_grad():
                model = models.predict_action().to(device=device)
                q_a = model(graph_input)
            max_position = (q_a == q_a[available].max().item()).nonzero()
            return torch.tensor(
                [random.choice(
                    np.intersect1d(available.cpu().contiguous().view(-1).numpy(),
                        max_position.cpu().contiguous().view(-1).numpy()))],
                dtype=torch.long)

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=6,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=31,
        scale=1.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    model.eval()
    model.to(device=device)

    env.reset()
    state = env.state
    state = np.array(state)
    graph = env.graph
    state_dim = graph.num_nodes
    act_dim = graph.num_nodes

    states = []
    actions = []
    rewards = []
    if mode == 'noise':
        state = env.state + np.random.normal(0, 0.1, size=env.state.shape)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []
    action_index_list = []
    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32).cpu() - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        for idx in np.argsort(action)[::-1]:
            if idx not in action_index_list:
                action_index = idx
                break

        action_index_list.append(action_index)
        reward, R, done = env.step(action_index)


        state[action_index] = 1
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - reward
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length, R
