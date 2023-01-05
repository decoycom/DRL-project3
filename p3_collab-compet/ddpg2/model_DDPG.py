import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor_DDPG(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[256,256,128],gate = F.relu,exit_gate = F.tanh):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first hidden layer
        """
        super(Actor_DDPG, self).__init__()
        
        self.gate = gate
        self.exit_gate = exit_gate
        self.seed = torch.manual_seed(seed)
        starter = [state_size]+fc_units
        stopper = fc_units+[action_size]
        self.fcs = []        
        for i in range(len(starter)):
            fc = nn.Linear(starter[i], stopper[i])
            setattr(self,"fc{}".format(i+1),fc)
            self.fcs.append(fc)
            
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)        
        # self.fc3 = nn.Linear(fc2_units, fc3_units)
        # self.fc4 = nn.Linear(fc3_units, action_size)
#         self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        for fc in self.fcs[:-1]:
            fc.weight.data.uniform_(*hidden_init(fc))            
        # self.fcs[-1].weight.data.uniform_(-3e-3, 3e-3)
        
        fc=self.fcs[-1]
        # for fc in self.fcs[:-1]:
        nn.init.orthogonal_(fc.weight.data)
        fc.weight.data.mul_(1e-3)
        nn.init.constant_(fc.bias.data, 0)
        
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # layer 3
        # self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        # self.fc4.weight.data.uniform_(-3e-3, 3e-3)        
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""        
        x = state
        for fc in self.fcs[:-1]:
            x = self.gate(fc(x))
        return self.exit_gate(self.fcs[-1](x))


class Critic_DDPG(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[256,256,128],gate = F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic_DDPG, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.gate = gate
        starter = [state_size+action_size]+fc_units
        stopper = fc_units+[1]
        
#         starter = [state_size]+[fc_units[0]+action_size]+fc_units[1:]
#         stopper = [fc_units[0]+action_size]+ fc_units[1:]+[1]
        
        self.fcs = []        
        for i in range(len(starter)):
            fc = nn.Linear(starter[i], stopper[i])
            setattr(self,"fc{}".format(i+1),fc)
            self.fcs.append(fc)
        
#         self.fcs1 = nn.Linear(state_size, fcs1_units)
#         self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)        
        # self.fc3 = nn.Linear(fc2_units, fc3_units)
        # self.fc4 = nn.Linear(fc3_units, 1)

#         self.fc3 = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for fc in self.fcs[:-1]:
            fc.weight.data.uniform_(*hidden_init(fc))            
        self.fcs[-1].weight.data.uniform_(-3e-3, 3e-3)
        
        fc=self.fcs[-1]
        # for fc in self.fcs[:-1]:
        nn.init.orthogonal_(fc.weight.data)
        fc.weight.data.mul_(1e-3)
        nn.init.constant_(fc.bias.data, 0)

        
        # self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        # self.fc4.weight.data.uniform_(-3e-3, 3e-3)
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat([state, action], dim=1)
        for fc in self.fcs[:-1]:
            x = self.gate(fc(x))
        
#         x = self.gate(self.fcs[0](state))
#         x = torch.cat((x, action), dim=1)
#         for fc in self.fcs[1:-1]:
#             x = self.gate(fc(x))
            
        return self.fcs[-1](x)
        
#         xs = F.relu(self.fcs1(state))
#         x = torch.cat((xs, action), dim=1)
#         x = F.relu(self.fc2(x))
#         # x = F.relu(self.fc3(x))
#         return self.fc3(x)
