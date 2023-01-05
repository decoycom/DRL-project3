import numpy as np
import ray
import ray.util.collective as col
@ray.remote(num_cpus=1,num_gpus=1)
class Worker:


    def setup(self, world_size, rank):
       self.rank = rank

       # self.group = "actor_i"
           # "actor_{}".format(rank)
       col.init_collective_group(world_size, rank, "gloo", self.group)
       # col.init_collective_group(world_size, rank, "nccl", "177")
       return True
    def __init__(self, config):
        self.state_size = config.state_dim
        self.action_size = config.action_dim
        self.rank = -1
        self.group = "actor_i"

        # [s,s,a,1,1]
        # state(s),next_state(s),action(a),reward(1),  done(1)
        self.state = np.ones((self.state_size * 2 + self.action_size + 1 + 1, ), dtype=np.float32)
        self.act = np.zeros((4, ), dtype=np.float32)

    def send_act(self):
        col.send(self.act, 0, self.group)

    def recv_state(self):
        col.recv(self.state, 0, self.group)
        return self.state

@ray.remote(num_cpus=1, num_gpus=0)
class Control:

    def setup(self, world_size, rank):
        self.rank = rank
        col.init_collective_group(world_size, rank, "gloo", self.group)
        # col.init_collective_group(world_size, rank, "nccl", "177")
        return True

    def __init__(self, config,actors):
        self.state_size = config.state_dim
        self.action_size = config.action_dim
        self.rank = -1
        self.actors = actors
        self.group = "actor_i"
        # self.group0 = "control_{}".format(0)

        # [s,s,a,1,1]
        # state(s),next_state(s),action(a),reward(1),  done(1)
        self.state = np.ones((self.actors,self.state_size * 2 + self.action_size + 1 + 1), dtype=np.float32)
        self.act = np.zeros((self.actors,4), dtype=np.float32)

    def recv_act(self,i):
        col.recv(self.act[i], 0, self.group)
        return self.act[i]

    def send_state(self,i):
        col.send(self.state[i], i, self.group)
        return

    def col_acts(self):
        for i in range(self.actors):
            self.recv_act(i)
        return self.act

    def send_states(self):
        for i in range(self.actors):
            self.send_state(i)

    # def allreduce_call(self):
    #    col.allreduce_multigpu([self.send1, self.send2], "177")
    #    return [self.send1, self.send2]
