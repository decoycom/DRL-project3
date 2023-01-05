
import numpy as np
import random
import torch
import torch.nn.functional as F

import sys


class TD3Ag():
    def __init__(self, config):
        self.config = config

        self.device = config.device
        self.state_size = config.state_dim
        self.action_size = config.action_dim

        self.network = config.network_fn(config)
        self.target_network = config.network_fn(config)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0

    def step(self, state, action, reward, next_state, done):
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

    def learn(self, exps):

        states, actions, rewards, next_states, mask = (torch.tensor(exps.state).float().to(self.config.device),
                                                       torch.tensor(exps.action).float().to(self.config.device),
                                                       torch.tensor(exps.reward).float().to(self.config.device), # .unsqueeze(-1)
                                                       torch.tensor(exps.next_state).float().to(self.config.device),
                                                       torch.tensor(exps.mask).float().to(self.config.device)) # .unsqueeze(-1)

        a_next = self.target_network(next_states)
        noise = torch.randn_like(a_next).mul(self.config.td3_noise)
        noise = noise.clamp(-self.config.td3_noise_clip, self.config.td3_noise_clip)

        min_a = float(-1.0)
        max_a = float(1.0)
        a_next = (a_next + noise).clamp(min_a, max_a)

        q_1, q_2 = self.target_network.q(next_states, a_next)
        target = rewards + self.config.discount * mask * torch.min(q_1, q_2)
        target = target.detach()

        q_1, q_2 = self.network.q(states, actions)
        critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

        self.network.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.config.gradient_clip)

        #torch.nn.utils.clip_grad_norm(self.network.fc_critic_1.parameters(), self.config.gradient_clip)
        #torch.nn.utils.clip_grad_norm(self.network.fc_critic_2.parameters(), self.config.gradient_clip)
        #torch.nn.utils.clip_grad_norm(self.network.critic_body_1.parameters(), self.config.gradient_clip)
        #torch.nn.utils.clip_grad_norm(self.network.critic_body_2.parameters(), self.config.gradient_clip)
        self.network.critic_opt.step()

        if self.total_steps % self.config.td3_delay:
            action = self.network(states)
            policy_loss = -self.network.q(states, action)[0].mean()

            self.network.zero_grad()
            policy_loss.backward()
            #torch.nn.utils.clip_grad_norm(self.network.fc_action.parameters(), self.config.act_clip)
            #torch.nn.utils.clip_grad_norm(self.network.actor_body.parameters(), self.config.act_clip)
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)

        self.soft_update(self.target_network, self.network)

    def soft_update(self, target, src):
        tau = self.config.target_network_mix
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - tau) +
                               tau * param)