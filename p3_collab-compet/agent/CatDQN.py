from collections import deque

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

import sys
def to_np(t):
    return t.cpu().detach().numpy()

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]

class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])

class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def compute_q(self, prediction):
        q_values = to_np(prediction['q'])
        return q_values

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        if config.noisy_linear:
            self._network.reset_noise()
        with config.lock:
            prediction = self._network(config.state_normalizer(self._state))
        q_values = self.compute_q(prediction)

        if config.noisy_linear:
            epsilon = 0
        elif self._total_steps < config.exploration_steps:
            epsilon = 1
        else:
            epsilon = config.random_action_prob()
        action = epsilon_greedy(epsilon, q_values)
        next_state, reward, done, info = self._task.step(action)
        entry = [self._state, action, reward, next_state, done, info]
        self._total_steps += 1
        self._state = next_state
        return entry

class CategoricalDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def _set_up(self):
        self.config.atoms = tensor(self.config.atoms)

    def compute_q(self, prediction):
        q_values = (prediction['prob'] * self.config.atoms).sum(-1)
        return to_np(q_values)


class CatDQNAg():
    def __init__(self, config):
        # state_size, action_size, seed

        self.config = config
        self.device = config.device
        self.state_size = config.state_dim
        self.action_size = config.action_dim
        
        self.config = config
        config.lock = mp.Lock()
        config.atoms = np.linspace(config.categorical_v_min,
                                   config.categorical_v_max, config.categorical_n_atoms)

        self.replay = config.replay_fn()
        self.actor = CategoricalDQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(config.batch_size)
        self.atoms = tensor(config.atoms)
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)

    def reset(self):
        self.random_process.reset_states()

    def step(self, states, actions, rewards, next_states, dones):

        # action = np.clip(action, -1.0, 1.0)

        # mask=1 - done,


        # d = dict(
        #     state = np.asarray([state], dtype=np.float32),
        #     action = np.asarray([action], dtype=np.float32),
        #     reward = np.asarray([reward], dtype=np.float32),
        #     next_state = np.asarray([next_state], dtype=np.float32),
        #     mask = 1 - np.asarray([done], dtype=np.int32),
        # )
        #
        # # print("m:{} r:{} a:{} s:{} n:{} ".format(d['mask'].shape,d['reward'].shape,d['action'].shape,d['state'].shape,d['next_state'].shape))
        # # print('step:',d)
        # # sys.stdout.flush()
        #
        # self.replay.feed(d)
        #
        # self.total_steps += 1
        #
        # if self.total_steps >= self.config.warm_up:
        #     # if self.config.eval_interval and not self.total_steps % self.config.eval_interval:
        #         transitions = self.replay.sample()
        #         self.learn(transitions)
        #

#         config = self.config

        # collect steps from agents
#         transitions = self.actor.step()
        for states, actions, rewards, next_states, dones, info in transitions:

            self.record_online_return(info)
            self.total_steps += 1
            self.replay.feed(dict(
                state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                action=actions,
                reward=[self.config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
            ))

        if self.total_steps > self.config.exploration_steps:
            transitions = self.replay.sample()
            if self.config.noisy_linear:
                self.target_network.reset_noise()
                self.network.reset_noise()
            loss = self.compute_loss(transitions)
            # if isinstance(transitions, PrioritizedTransition):
            #     priorities = loss.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
            #     idxs = tensor(transitions.idx).long()
            #     self.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
            #     sampling_probs = tensor(transitions.sampling_prob)
            #     weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.config.replay_beta())
            #     weights = weights / weights.max()
            #     loss = loss.mul(weights)

            loss = self.reduce_loss(loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with self.config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def compute_loss(self, transitions):
        config = self.config
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)
        with torch.no_grad():
            prob_next = self.target_network(next_states)['prob']
            q_next = (prob_next * self.atoms).sum(-1)
            if config.double_q:
                a_next = torch.argmax((self.network(next_states)['prob'] * self.atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[self.batch_indices, a_next, :]

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        atoms_target = rewards + self.config.discount ** config.n_step * masks * self.atoms.view(1, -1)
        atoms_target.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * \
                      prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = tensor(transitions.action).long()
        log_prob = log_prob[self.batch_indices, actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def reduce_loss(self, loss):
        return loss.mean()            
            
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
        # torch.nn.utils.clip_grad_norm(self.network.critic.parameters(), 1)
        self.network.critic_opt.step()

        # Actor
        # phi = self.network.feature(states).float()
        action_pred = self.network.actor(states)
        # Make it ascend
        policy_loss = -self.network.critic(states, action_pred).mean()

        self.network.zero_grad()
        policy_loss.backward()
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
        critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

        self.network.zero_grad()
        critic_loss.backward()
        # Clip
        torch.nn.utils.clip_grad_norm(self.network.critic_params, 1)
        # torch.nn.utils.clip_grad_norm(self.network.fc_critic.parameters(), 1)
        # torch.nn.utils.clip_grad_norm(self.network.fc_critic.parameters(), 1)
        self.network.critic_opt.step()

        phi = self.network.feature(states)
        action = self.network.actor(phi)
        policy_loss = -self.network.critic(phi.detach(), action).mean()

        self.network.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        self.soft_update(self.target_network, self.network)

    def soft_update(self, target, src):
        tau = self.config.target_network_mix
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - tau) +
                               tau * param)