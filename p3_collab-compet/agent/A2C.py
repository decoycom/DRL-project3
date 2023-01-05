import numpy as np
import random
import torch
import torch.nn.functional as F

import sys

def to_np(t):
    return t.cpu().detach().numpy()

class A2CAg():
    def __init__(self, config):
        # BaseAgent.__init__(self, config)
        self.config = config

        self.device = config.device

        self.state_size = config.state_dim
        self.action_size = config.action_dim
        
        self.gamma = 1.0 - (0.01 * config.discount)
        self.gae_tau = 1.0 - (0.01 * config.gae_tau)

        self.network = config.network_fn(config)
        self.optimizer = config.optimizer_fn(self.network.parameters(),config)
        self.total_steps = 0
        # self.states = self.task.reset()

    def initState(self, states):
        self.states = states

    def act(self, states):
        # print("states.shape:{}".format(states.shape),  flush = True )
        actions = self.network(self.config.state_normalizer(states))
        # print("actions.shape:{}".format(actions['action'].shape), flush=True)
        # print("actions:{}".format(actions), flush=True)
        return actions
    
    def filename(self, it,s_postfix,score):
        st = ''
        st += 'f1-{}'.format(  int( getattr(self.config,'fc1',1)))
        st += '_f2-{}'.format( int( getattr(self.config,'fc2',1)))
        st += '_f3-{}'.format( int( getattr(self.config,'fc3',1)))
        st += '_al-{}'.format( int( getattr(self.config,'act_layers',1)))
        st += '_cl-{}'.format( int( getattr(self.config,'crt_layers',1)))
        return 'checkpoint_s-{:.1f}_{}_{}_{}_a2c.pth'.format(score,it,st,s_postfix)
    
    def save(self, file):
        torch.save(self.network.state_dict(), file)
        
    def load(self, file):
        self.network.load_state_dict(torch.load(file))

    def collect(self, states, step_fn):
        storage = self.config.storage_fn()
        rews = np.zeros(self.config.num_workers)
        ln = 0
        for _ in range(self.config.rollout_length):
            try:
                prediction = self.act(states)
                next_states, rewards, terminals, info = step_fn(to_np(prediction['action']))
                rewards = np.nan_to_num(rewards, nan=-1.0)
                rewards = self.config.reward_normalizer(rewards)
                rews += rewards
                ln +=1
            except:
                rewards = np.zeros(self.config.num_workers)
                terminals = np.ones(self.config.num_workers)
#                 np.zeros(self.config.num_workers)
#                 next_states, rewards, terminals, info = step_fn(np.random.randn(self.config.num_workers, self.action_size))
#                 terminals[0] = 1.0
#                 states = np.random.randn(self.config.num_workers, self.state_size)
                rewards = np.nan_to_num(rewards, nan=-1.0)
                rewards = self.config.reward_normalizer(rewards)
                break
            next_states = np.nan_to_num(next_states)
            storage.feed(prediction)
            storage.feed({  'reward': torch.tensor(rewards).unsqueeze(-1).to(self.config.device),
                            'mask': torch.tensor(1 - terminals).unsqueeze(-1).to(self.config.device)})
            states = next_states
            self.total_steps += self.config.num_workers

        return (storage,states,rews,terminals,ln)

    # step rollout_length
    def learn(self, states, step_fn):
        config = self.config
        states = np.nan_to_num(states)
        # torch.nan_to_num(torch.tensor(states).float(), nan=0.0)
        storage,states,rewards,dones,ln = self.collect(states, step_fn)
#         if np.any(dones):
#             return states,rewards,dones
        try:
            prediction = self.network(self.config.state_normalizer(states))
        except:
            return states,rewards,dones
        storage.feed(prediction)
        storage.placeholder()

        advantages = torch.tensor(np.zeros((config.num_workers, 1))).to(self.config.device)

        tau_dis = torch.tensor(self.gae_tau * self.gamma).to(self.config.device)

        returns = prediction['v'].detach()  #.cpu()
        for i in reversed(range(ln)):  # config.rollout_length
            returns = storage.reward[i] + self.gamma * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + self.gamma * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * tau_dis * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        try:
            entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
            policy_loss = -(entries.log_pi_a * entries.advantage).mean()
            value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
            entropy_loss = entries.entropy.mean()

            self.optimizer.zero_grad()
            (policy_loss - config.entropy_weight * entropy_loss +
             config.value_loss_weight * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
            self.optimizer.step()
        except:
            return states,rewards,dones
        
        return states,rewards,dones