import numpy as np
import torch
import torch.nn.functional as F

import time




class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, state_dim, act_dim, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        eval_start = time.time()
        self.model.state_dim = state_dim
        self.model.act_dim = act_dim
        self.model.train()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def train_step1(self, states, state_dim, actions, rewards, returns_to_go, timesteps, **kwargs):

        states = states.reshape(1, -1, self.model.state_dim)
        actions = actions.reshape(1, -1, self.model.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.model.max_length is not None:
            states = states[:,-self.model.max_length:]
            actions = actions[:,-self.model.max_length:]
            returns_to_go = returns_to_go[:,-self.model.max_length:]
            timesteps = timesteps[:,-self.model.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.model.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.model.max_length-states.shape[1], self.model.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.model.max_length - actions.shape[1], self.model.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.model.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.model.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None


        _, action_preds, return_preds = self.model.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
        # print(action_preds, return_preds)
        output_action = action_preds[0, -1]
        logits = self.model.fc(output_action)
        probs = F.softmax(logits, dim=-1)
        max_prob, max_index = torch.max(probs, dim=-1)
        l = len(rewards)
        a = []
        for i in range(2, l + 1):
            a.append(actions[:, -i])
        for h in a:
            h1 = h.item()
            h1 = int(h1)
            probs[h1] = 0.0
        if max_index in actions:
            max_prob1, max_index1 = torch.max(probs, dim=-1)
            return max_index1

        return max_index





