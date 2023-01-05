import numpy as np
import random
import copy
from collections import namedtuple, deque
from .model_DDPG import Actor_DDPG, Critic_DDPG

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.98            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.008    # L2 weight decay

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_DDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.config = config
        self.device = config.device
        device = config.device

        state_size = config.state_dim
        action_size = config.action_dim
        self.batch_size = 64*config.batch_size
        
        random_seed = config.seed
        fc1 = 64*config.fc1
        fc2 = int(fc1 / config.fc2)
        fc3 = int(fc2 / config.fc3)
        fcs = [fc1,fc2,fc3]
        
        self.weight_decay_act = config.weight_decay_act
        self.weight_decay = config.weight_decay
        self.rollout = config.eval_interval
        
        # config.actor_lr = 1e-3
        # config.critic_lr = 1e-3
        self.gamma = 1.0 - (0.01 * config.discount)
        # = 0.98
        
        self.tau = 1e-5 * config.target_network_mix
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        actor_lr = 1e-5 * config.actor_lr
        critic_lr = 1e-5 * config.critic_lr

        self.std = nn.Parameter(torch.zeros(action_size))
        
        # Actor Network (w/ Target Network)
        # , fc3_units=fc3
        self.actor_local = Actor_DDPG(state_size, action_size, random_seed,fc_units=fcs[:config.act_layers]).to(self.device)
        self.actor_target = Actor_DDPG(state_size, action_size, random_seed,fc_units=fcs[:config.act_layers]).to(self.device)
        
        self.actor_params = list(self.actor_local.parameters())
        self.actor_params.append(self.std)
        self.actor_optimizer = optim.Adam(self.actor_params, lr=actor_lr, weight_decay=self.weight_decay_act)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic_DDPG(state_size, action_size, random_seed,fc_units=fcs[:config.crt_layers]).to(self.device)
        self.critic_target = Critic_DDPG(state_size, action_size, random_seed,fc_units=fcs[:config.crt_layers]).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr, weight_decay=self.weight_decay)

        
        # Noise process
        # config.n_mu
        self.noise = OUNoise(action_size, random_seed,mu=0., theta=config.n_theta, sigma=config.n_sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.batch_size*config.memory_size, self.batch_size, random_seed,device=config.device)
        self.total_steps = 0
        # self.rollout = 5
    
    def save(self, it,s_postfix):
        torch.save(self.actor_local.state_dict(), 'checkpoint_{}{}_actor.pth'.format(it,s_postfix))
        torch.save(self.critic_local.state_dict(), 'checkpoint_{}{}_critic.pth'.format(it,s_postfix))

    def load(self, it,s_postfix):
        self.actor_local.load_state_dict(torch.load('checkpoint_{}{}_actor.pth'.format(it,s_postfix)))
        self.critic_local.load_state_dict(torch.load('checkpoint_{}{}_critic.pth'.format(it,s_postfix)))
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.total_steps += 1
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            if self.total_steps % self.rollout == 0:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        # if self.total_steps < self.config.warm_up:
        #     action = np.random.randn(1, self.action_size)
        #     # np.random.randn(self.action_size)
        # else:
        state = torch.from_numpy(state).float().to(self.device)

        # self.actor_local.eval()
        # with torch.no_grad():
        mean = self.actor_local(state).detach().cpu()
        # .cpu().numpy() 
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        action = dist.sample()

        # action = self.actor_local(state)
        action = action.cpu().numpy()                
        # self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # critic_loss = F.mse_loss(Q_expected, Q_targets)
        # .mean()
        
        # critic_loss = F.mse_loss(Q_expected, Q_targets).mul(0.5).sum(-1).mean()
        critic_loss = (Q_expected - Q_targets).pow(2).mul(0.5).sum(-1).mean()
        
        # Minimize the loss
        # self.critic_local.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_target.parameters(), self.config.gradient_clip)
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        # self.actor_local.zero_grad()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.config.act_clip)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.detach_()
            target_param.copy_(tau*local_param + (1.0-tau)*target_param)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed,device='cpu'):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)