import numpy as np
import random
import torch
import torch.nn.functional as F

import sys

def to_np(t):
    return t.cpu().detach().numpy()

class DDPGAg():
    def __init__(self, config):
        # state_size, action_size, seed

        self.config = config

        self.device = config.device

        self.state_size = config.state_dim
        self.action_size = config.action_dim

        self.network = config.network_fn(config)
        self.target_network = config.network_fn(config)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(config)
        self.total_steps = 0
        self.random_process.reset_states()

    def reset(self):
        self.random_process.reset_states()

    def step(self, state, action, reward, next_state, done):

        # action = np.clip(action, -1.0, 1.0)

        # mask=1 - done,


        d = dict(
            state = np.asarray([state], dtype=np.float32),
            action = np.asarray([action], dtype=np.float32),
            reward = np.asarray([reward], dtype=np.float32),
            next_state = np.asarray([next_state], dtype=np.float32),
            mask = 1 - np.asarray([done], dtype=np.int32),
        )

        # print("m:{} r:{} a:{} s:{} n:{} ".format(d['mask'].shape,d['reward'].shape,d['action'].shape,d['state'].shape,d['next_state'].shape))
        # print('step:',d)
        # sys.stdout.flush()

        self.replay.feed(d)

        self.total_steps += 1

        if self.total_steps >= self.config.warm_up:
            # if self.config.eval_interval and not self.total_steps % self.config.eval_interval:
                transitions = self.replay.sample()
                self.learn(transitions)

# TODO make it multi action
    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # actions = np.random.randn(num_agents, self.action_size)

        # print("state:{}".format(state))


        if self.total_steps < self.config.warm_up:
            action = np.random.randn(self.action_size)

            # print('act action:', torch.from_numpy(action).shape)
            # sys.stdout.flush()
            # action = random.choice(np.arange(self.action_size))
            # action = actions[0]
        else:
            state = torch.from_numpy(state).float()
            # print('act new state1:', state.shape)

            state= state.to(self.device)
            # Make an array
            # .unsqueeze(0)
            # print('act new state2:', state.shape)

            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()

            action = action_values.cpu().data.numpy()
            # noise
            action += self.random_process.sample()

            # print('act new action:', torch.from_numpy(action).shape)
            # sys.stdout.flush()
        return np.clip(action, -1, 1)

    def learn2(self, exps):

        # print("exps:{}".format(exps))
        # sys.stdout.flush()

        states, actions, rewards, next_states, mask = (torch.tensor(exps.state).float().to(self.config.device),
                                                       torch.tensor(exps.action).float().to(self.config.device),
                                                       torch.tensor(exps.reward).float().to(self.config.device), # .unsqueeze(-1)
                                                       torch.tensor(exps.next_state).float().to(self.config.device),
                                                       torch.tensor(exps.mask).float().to(self.config.device)) # .unsqueeze(-1)

        # Target
        # phi_next = self.target_network.feature(next_states).float()
        a_next = self.target_network.actor(next_states)
        q_next = self.target_network.critic(next_states, a_next)

        # apply Gamma(discount)
        q_next = self.config.discount * mask * q_next

        # Q_Targets = rewards + (gamma * Q_targets_next * (1 - dones))
        q_next.add_(rewards)
        q_next = q_next.detach()

        # Local Q_expected
        # phi = self.network.feature(states).float()
        q = self.network.critic(states, actions.squeeze(1))

        # critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
        critic_loss = F.mse_loss(q, q_next)

        self.network.zero_grad()
        critic_loss.backward()
        # Clip
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.config.gradient_clip)
        # torch.nn.utils.clip_grad_norm(self.network.fc_critic.parameters(), self.config.gradient_clip)
        # torch.nn.utils.clip_grad_norm(self.network.critic_body.parameters(), self.config.gradient_clip)
        self.network.critic_opt.step()

        # Actor
        # phi = self.network.feature(states).float()
        action_pred = self.network.actor(states)
        # Make it ascend
        policy_loss = -self.network.critic(states, action_pred).mean()

        self.network.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.network.fc_action.parameters(), self.config.act_clip)
        # torch.nn.utils.clip_grad_norm(self.network.actor_body.parameters(), self.config.act_clip)
        self.network.actor_opt.step()

        # update target
        self.soft_update(self.target_network, self.network)

    def learn(self, exps):

        states, actions, rewards, next_states, mask = (torch.tensor(exps.state).float().to(self.config.device),
                                                       torch.tensor(exps.action).float().to(self.config.device),
                                                       torch.tensor(exps.reward).float().to(self.config.device), # .unsqueeze(-1)
                                                       torch.tensor(exps.next_state).float().to(self.config.device),
                                                       torch.tensor(exps.mask).float().to(self.config.device)) # .unsqueeze(-1)


        phi_next = self.target_network.feature(next_states)
        a_next = self.target_network.actor(phi_next)
        q_next = self.target_network.critic(phi_next, a_next)

        # mask = 1-np.asarray(dones, dtype=np.int32)
        # apply Gamma(discount)
        q_next = self.config.discount * mask * q_next

        q_next.add_(rewards)
        q_next = q_next.detach()
        phi = self.network.feature(states)
        q = self.network.critic(phi, actions)
        
        # critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
        critic_loss = F.mse_loss(q, q_next)

        self.network.zero_grad()
        critic_loss.backward()
        # Clip
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.config.gradient_clip)
        # torch.nn.utils.clip_grad_norm(self.network.critic_params, 1)
        # torch.nn.utils.clip_grad_norm(self.network.fc_critic.parameters(), 1)
        # torch.nn.utils.clip_grad_norm(self.network.fc_critic.parameters(), self.config.gradient_clip)
        # torch.nn.utils.clip_grad_norm(self.network.critic_body.parameters(), self.config.gradient_clip)
        self.network.critic_opt.step()

        phi = self.network.feature(states)
        action = self.network.actor(phi)
        policy_loss = -self.network.critic(phi.detach(), action).mean()

        self.network.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.network.fc_action.parameters(), self.config.gradient_clip)
        # torch.nn.utils.clip_grad_norm(self.network.actor_body.parameters(), self.config.gradient_clip)
        self.network.actor_opt.step()

        self.soft_update(self.target_network, self.network)

    def soft_update(self, target, src):
        tau = self.config.target_network_mix
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - tau) +
                               tau * param)