import numpy as np
import pandas as pd
import torch
from time import sleep

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from deep_rl import *
from agent import *
from ddpg2 import *

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from joblib import Parallel, delayed

import os
base_path = os.getcwd()

# === [ Brains ] ===

def brain(brain_name,agent,config,env, it=0, fig = None, ax = None,log_prefix='brain_'):
    n_episodes = config.eval_episodes # 2000, 
    max_t = config.max_steps # 1500, 
    # eps_start=1.0, 
    # eps_end=0.01, 
    # eps_decay=0.995
    window = getattr(config, 'scores_window', 100)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=window)  # last 100 scores
    scores_window_mean = []
    # eps = eps_start                    # initialize epsilon

    pritty = getattr(config, 'pritty_fields', {})
    st = ''
    for k in config.update_fields:
        v = getattr(config, k)
        if k in pritty:
            k = pritty[k]
        if float(v).is_integer():
            st += '{}:{} '.format(k,v)
        else:
            st += '{}:{:.5f} '.format(k,v)
    # print('\rStart[{}]\t{}'.format(it,st), flush = True)
    log("Brain [{}] {}".format(it,st),log_prefix)

    no_reg = getattr(config, 'stop_regression', True)
    max_reg = getattr(config, 'max_regression', 0.20)
    perc_reg = getattr(config, 'perc_regression', 20)
    s_margin = getattr(config, 'save_margin', 32.0)
    s_postfix = getattr(config, 'save_postfix', '')    
    win_mean = 0
    
    starter = -10000
    max_win = starter
    last_win = win_mean
    max_mean = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations
        agent.reset()
        
        score = 0
        for t in range(max_t):
            action = agent.act(state) # eps
            
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
    
            agent.step(state[0], action[0], reward, next_state[0], done)
        
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        
        win_mean = np.mean(scores_window)
        scores_window_mean.append(win_mean)
        if max_win != starter:
            max_mean = max(max_mean,win_mean)
        if fig is not None and ax is not None:
            draw(fig,ax,[scores,scores_window_mean],'e:{} win:{:.2f} max:{:.2f}\n{}'.format(it,win_mean, max_mean,st))

        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        # print('\rEpisode[{}]\t{} a-Average Score: {:.2f} max.win.mean:{:.2f}'.format(it, i_episode, win_mean, max_mean), end="")
        if i_episode % window == 0:
            if no_reg and max_win - (max_win / 100.0 * perc_reg) > win_mean :
                log('=[Avg:{:.2f} max:{:.2f}]= Finished![{}] {}\t{} '.format(win_mean, max_mean ,it,st, i_episode),log_prefix)
                print('\rFinished![{}] {}\t{} a-Average Score: {:.2f} max.win.mean:{:.2f}'.format(it,st, i_episode ,win_mean, max_mean), flush = True)
                return (scores, win_mean,last_win >= s_margin,max_win,max_mean)
            else:
                log('=[Avg:{:.2f} max:{:.2f}]= Episode[{}]\t{}'.format(win_mean, max_mean,it, i_episode),log_prefix)
            max_win = max(max_win, win_mean)
            max_mean = max(max_mean,max_win)
    
        if win_mean >= s_margin:
            if last_win < win_mean: 
                log('=[Avg:{:.2f} max:{:.2f}]= Episode[{}] Solved in {:d} episodes!\t{}'.format(win_mean, max_mean,it,i_episode-window,st),log_prefix)
                agent.save(agent.filename(it,s_postfix,win_mean))
                last_win = win_mean

    return (scores, win_mean,last_win >= s_margin,max_win,max_mean)

#                (brain_name,agent,config,env, it=0, fig = None, ax = None,log_prefix='brain_')
def brain_multy(brain_name,agent,config,env, it=0, fig = None, ax = None,log_prefix='brain_'):
    number = config.num_workers
    n_episodes = config.eval_episodes # 2000, 
    max_t = config.max_steps # 1500, 
    window = getattr(config, 'scores_window', 100)
    
    scores = []                        # list containing scores from each episode
    scores_window_mean = []
    scores_window = []  # last 100 scores
    
    for i in range(number):
#         scores.append([])
        scores_window.append(deque(maxlen=window))
    
    # eps = eps_start                    # initialize epsilon

    pritty = getattr(config, 'pritty_fields', {})
    st = ''
    for k in config.update_fields:
        v = getattr(config, k)
        if k in pritty:
            k = pritty[k]
        if float(v).is_integer():
            st += '{}:{} '.format(k,v)
        else:
            st += '{}:{:.5f} '.format(k,v)
    # print('\rStart[{}]\t{}'.format(it,st), flush = True)
    log("Brain [{}] {}".format(it,st),log_prefix)

#     next_states, rewards, terminals, info
    def step_fn(actions):
        env_info = env.step(actions)[brain_name]        # send the action to the environment
        next_states = env_info.vector_observations   # get the next state
        rewards = env_info.rewards                   # get the reward
        dones = env_info.local_done                  # see if episode has finished
        dones = np.asarray(dones, dtype=np.int32)
        return (next_states,rewards,dones,env_info)
        
#     = lambda action: torch.optim.Adam(params, lr=config.actor_lr), #, weight_decay=config.weight_decay

    no_reg = getattr(config, 'stop_regression', True)
    max_reg = getattr(config, 'max_regression', 0.5)
    perc_reg = getattr(config, 'perc_regression', 20)
    s_margin = getattr(config, 'save_margin', 32.0)
    s_postfix = getattr(config, 'save_postfix', '')
    
    win_mean = 0
   
    starter = -10000
    max_win = starter
    last_win = starter
    max_mean = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations
        score = np.zeros(number)
        for t in range(max_t):
            states,rewards,dones = agent.learn(states,step_fn)         
            score = score + rewards
            if np.any(dones):
                break 
        for i in range(number):
            scores_window[i].append(score[i])       # save most recent score
        scores.append(np.mean(score))              # save most recent score
        win_mean = np.mean(scores_window)
        scores_window_mean.append(win_mean)
        if max_win != starter:
            max_mean = max(max_mean,win_mean)
        if fig is not None and ax is not None:
            draw(fig,ax,[scores,scores_window_mean],'e:{} win:{:.2f} max:{:.2f}\n{}'.format(it,win_mean, max_mean,st))

        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        # print('\rEpisode[{}]\t{} a-Average Score: {:.2f} max.win.mean:{:.2f}'.format(it, i_episode, win_mean, max_mean), end="")
        if i_episode % window == 0:
            if no_reg and max_win - (max_win / 100.0 * perc_reg) > win_mean :
                log('=[Avg:{:.2f} max:{:.2f}]= Finished![{}] {}\t{} '.format(win_mean, max_mean ,it,st, i_episode),log_prefix)
                print('\rFinished![{}] {}\t{} a-Average Score: {:.2f} max.win.mean:{:.2f}'.format(it,st, i_episode ,win_mean, max_mean), flush = True)
                return (scores, win_mean,last_win >= s_margin,max_win,max_mean)
            else:
                log('=[Avg:{:.2f} max:{:.2f}]= Episode[{}]\t{}'.format(win_mean, max_mean,it, i_episode),log_prefix)
            max_win = max(max_win, win_mean)
            max_mean = max(max_mean,max_win)
    
        if win_mean >= s_margin:
            if last_win < win_mean: 
                log('=[Avg:{:.2f} max:{:.2f}]= Episode[{}] Solved in {:d} episodes!\t{}'.format(win_mean, max_mean,it,i_episode-window,st),log_prefix)
                agent.save(agent.filename(it,s_postfix,win_mean))
                last_win = win_mean

    return (scores, win_mean,last_win >= s_margin,max_win,max_mean)

# === [ Strategies ] ===

def initConf_ddpg(state_size,action_size,brain_name,env):
    # select_device(0)

    config = Config()

    config.update_fields = [
        'fc1','fc2',
        'r_proc',
        'weight_decay_act','weight_decay',
        'actor_lr','critic_lr',
        'target_network_mix','discount',
        'gradient_clip',
        # 'act_clip',
    ]
    config.pritty_fields = {'weight_decay_act':'a_W','weight_decay':'c_W','actor_lr':'a_Lr','critic_lr':'c_Lr','target_network_mix':'tau','gradient_clip':'c_Clip','act_clip':'a_Clip','discount':'G'}
    # config.merge(kwargs)
    config.device = Config.DEVICE
    # print("config.device:{}".format(config.device))
    config.brain_name = brain_name
    config.seed = 0    

    config.fc1 = 300
    config.fc2 = 200
    config.fc3 = 100

    config.eval_episodes = 200
    config.max_steps = int(1e7)
    config.batch_size = 64
    
    config.gradient_clip = 5
    config.eval_interval = 4
    config.memory_size = int(1e6)            
    config.warm_up = config.batch_size
    config.r_proc = 0.2

    # num_agents
    config.state_dim = state_size
    config.action_dim = action_size
    # 400, 300
    config.network_fn = lambda cfg: DeterministicActorCriticNet(
        cfg.state_dim, cfg.action_dim,
        actor_body=FCBody(cfg.state_dim, (cfg.fc1, cfg.fc2), gate=F.relu),
        critic_body=FCBody(cfg.state_dim + cfg.action_dim, (cfg.fc1, cfg.fc2), gate=F.relu),

        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=cfg.actor_lr, weight_decay=cfg.weight_decay_act),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=cfg.critic_lr, weight_decay=cfg.weight_decay))

    # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)    
    config.replay_fn = lambda: UniformReplay(memory_size=config.memory_size, batch_size=config.batch_size)

    config.agent_fn = lambda conf: DDPGAg(conf)
    config.brain_fn = lambda cfg, **kwargs: brain(cfg.brain_name,cfg.agent_fn(cfg),cfg,env,**kwargs)

    config.random_process_fn = lambda cfg: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(cfg.r_proc)) 
    return config


def initConf_ddpg2(state_size,action_size,brain_name,env):
    # select_device(0)

    config = Config()

    config.update_fields = [
        'fc1','fc2','fc3',
        'act_layers','crt_layers',
        'weight_decay_act','weight_decay',
        'actor_lr','critic_lr',
        'target_network_mix','discount',
        'gradient_clip','act_clip',
        'n_mu','n_theta','n_sigma',
        'memory_size','batch_size',
    ]
    config.pritty_fields = {'weight_decay_act':'a_W','weight_decay':'c_W','actor_lr':'a_Lr','critic_lr':'c_Lr','target_network_mix':'tau','gradient_clip':'c_Clip','act_clip':'a_Clip','discount':'G'}
    # config.merge(kwargs)
    config.device = Config.DEVICE
    # print("config.device:{}".format(config.device))
    config.brain_name = brain_name
    config.seed = 0    

    config.n_mu = 0.0
    config.n_theta=0.15
    config.n_sigma=0.2

    config.fc1 = 4 # 256
    config.fc2 = 1 # 256 / 1
    config.fc3 = 2 # 256 / 1 / 2
    config.weight_decay_act = 0.0001
    config.weight_decay = 0.0001
    config.actor_lr = 0.0001
    config.critic_lr = 0.0001
    config.target_network_mix = 1e-3
    config.discount = 0.99
    config.gradient_clip = 7.0
    config.act_clip = 7

    config.eval_episodes = 200
    config.max_steps = int(4e12)
    config.batch_size = 1
    
    config.gradient_clip = 5
    config.eval_interval = 8
    config.memory_size = int(100)            
    config.warm_up = config.batch_size *64 * 4
    
    config.act_layers = 1
    config.crt_layers = 3
    # config.rollout = 10

    # num_agents
    config.state_dim = state_size
    config.action_dim = action_size

    config.agent_fn = lambda conf: Agent_DDPG(conf)
    config.brain_fn = lambda cfg, **kwargs: brain(cfg.brain_name,cfg.agent_fn(cfg),cfg,env, **kwargs)

    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2)) 
    return config

def initConf_td3(state_size,action_size,brain_name,env):
    
    # select_device(0)
    config = Config()
    config.brain_name = brain_name

    config.update_fields = ['fc1','fc2','fc3','weight_decay','weight_decay_act','actor_lr','critic_lr','target_network_mix','discount','gradient_clip','td3_noise','td3_noise_clip','td3_delay']
    config.pritty_fields = {'weight_decay_act':'a_W','weight_decay':'c_W','actor_lr':'a_Lr','critic_lr':'c_Lr','target_network_mix':'tau','gradient_clip':'c_Clip','act_clip':'a_Clip','discount':'G', 'td3_noise':'N','td3_noise_clip':'N_Clip','td3_delay':'Delay'}
    #,'fc3'
    config.fc1 = 300
    config.fc2 = 200
    config.fc3 = 100

    config.device = Config.DEVICE
    config.state_dim = state_size
    config.action_dim = action_size

    config.max_steps = int(1e12)
    config.eval_episodes = 128
    config.batch_size = 128

    config.eval_interval = int(1e4)
    config.memory_size=int(1e5)

    config.gradient_clip = 5
    config.act_clip = 9
    config.weight_decay=0.012
    config.actor_lr = 1e-4
    config.critic_lr = 1e-4
    config.target_network_mix = 5e-3
    config.discount = 0.98

    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2

    config.network_fn = lambda cfg: TD3Net(
        cfg.action_dim,
        actor_body_fn=lambda: FCBody(cfg.state_dim, (cfg.fc1, cfg.fc2, cfg.fc3), gate=F.relu),
        critic_body_fn=lambda: FCBody(cfg.state_dim + cfg.action_dim, (cfg.fc1, cfg.fc2, cfg.fc3), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=cfg.actor_lr, weight_decay=cfg.weight_decay_act),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=cfg.critic_lr,weight_decay=cfg.weight_decay))
    #     .to(config.device)

    replay_kwargs = dict(
        memory_size=config.memory_size,
        batch_size=config.batch_size,
    )

    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async=False)
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))

    config.warm_up = int(1e4)

    config.agent_fn = lambda conf: TD3Ag(conf)
    config.brain_fn = lambda cfg, **kwargs: brain(cfg.brain_name,cfg.agent_fn(cfg),cfg,env, **kwargs)
    
    return config

def a2c_feature(state_size,action_size,brain_name,env):
    config = Config()
    config.brain_name = brain_name

    config.update_fields = [
        'fc1','fc2','fc3',
        'act_layers','crt_layers',
        'discount','gradient_clip',
        'rollout_length', 'lr', 
        'use_gae','gae_tau', 'entropy_weight']
    config.pritty_fields = {'rollout_length':'rout_L','gae_tau':'G_tau','entropy_weight':'e_W','use_gae':'G_use','gradient_clip':'Clip','discount':'G'}
    config.max_steps = int(2e7)

#     config.eval_interval = int(1e4)
    config.eval_episodes = 400
    config.batch_size = 128
#     config.memory_size=int(1e6)
    
    config.device = Config.DEVICE
    config.state_dim = state_size
    config.action_dim = action_size
    # {'rollout_length': 8, 'discount': 0.9751201238174458, 'gradient_clip': 7, 'lr': 1.0802107431577493e-05, 'gae_tau': 0.9964673379102222, 'entropy_weight': 0.0015039861493695169, 'use_gae': False}
    # {'rollout_length': 10, 'discount': 0.9847894774762507, 'gradient_clip': 2, 'lr': 1.0071160478803324e-05, 'gae_tau': 0.9899540738644809, 'entropy_weight': 0.0011836478393662721, 'use_gae': True},
    config.discount = 0.9751201238174458
    config.use_gae = False
    config.gae_tau = 0.9964673379102222
    config.entropy_weight = 0.0015039861493695169
    config.rollout_length = 8
    config.gradient_clip = 7
    config.lr = 1.0802107431577493e-05

    config.num_workers = 20
    
    config.fc1 = 6
    config.fc2 = 1
    config.fc3 = 2
    
    config.act_layers = 1
    config.crt_layers = 3
    
    config.optimizer_fn = lambda params,cfg: torch.optim.RMSprop(params, lr= 1e-5 * cfg.lr)
    config.network_fn = lambda cfg: GaussianActorCriticNet(
        cfg.state_dim, cfg.action_dim,
        actor_body=FCBody(cfg.state_dim, 
                          tuple(e for e in [64*cfg.fc1,64*cfg.fc1 // cfg.fc2 ,64*cfg.fc1 // cfg.fc2 // cfg.fc3][:cfg.act_layers])),
        critic_body=FCBody(cfg.state_dim, 
                           tuple(e for e in [64*cfg.fc1,64*cfg.fc1 // cfg.fc2 ,64*cfg.fc1 // cfg.fc2 // cfg.fc3][:cfg.crt_layers])))
        
#     config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
#     config.network_fn = lambda: CategoricalActorCriticNet(
#         config.state_dim, config.action_dim, FCBody(config.state_dim, gate=F.tanh))
    
    config.storage_fn = lambda:  Storage(config.rollout_length)
    
    config.agent_fn = lambda cfg: A2CAg(cfg)
    config.brain_fn = lambda cfg, **kwargs: brain_multy(cfg.brain_name,cfg.agent_fn(cfg),cfg,env,**kwargs)
    
    return config

def log(text,prefix='pre_',file='log.log'):
    with open("{}log.log".format(prefix), "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("\n{}".format(text))

def log_reset(prefix='pre_',file='log.log'):
    f = "{}log.log".format(prefix)
    with open(f, "w") as file_object:
        file_object.write("Started {}!".format(f))
        
# === [ Tuners ] ===

def draw(fig,ax,scoresz,title = 'Title'):
    ax.cla()
    for scores in scoresz:
        ax.plot(np.arange(len(scores)),scores)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title(title) # , fontsize=next(fontsizes)
    display(fig)
    clear_output(wait = True)
    plt.pause(0.001)    

def tune(
    config : Config,
    space_cfg, 
    max_iter=10, 
    hebo_cfg = None,
    greater_is_better : bool = True,
    verbose  = False,
    first = [],
    fig = None,
    ax = None,
    log_prefix = 'tune_',
    **kwargs
    ):
    log_reset(log_prefix)
    if hebo_cfg is None:
        hebo_cfg = {}
    space = DesignSpace().parse(space_cfg)
    opt   = HEBO(space, **hebo_cfg)    
    
    scoresz = []
    first_index = 0
    for i in range(max_iter):
        if len(first) > first_index:
            hyp = first[first_index]
            first_index += 1
            details = {}
            for n,v in hyp.items():
                details[n] = [v]
            rec = pd.DataFrame(details)
        else:
            rec     = opt.suggest()
            hyp     = rec.iloc[0].to_dict()
        for k in config.update_fields:
#         for k in hyp:
            if space.paras[k].is_numeric and space.paras[k].is_discrete:
#                 hyp[k] = int(hyp[k])
                setattr(config, k, int(hyp[k]))
            else:
                setattr(config, k, hyp[k])

        scores,reward,done,max_win,max_mean = config.brain_fn(config,it=i,fig=fig,ax=ax,log_prefix=log_prefix)
        # brain(brain_name,a,config,env)
        scoresz.append(scores)
        
        if fig is not None and ax is not None:
            draw(fig,ax,[scores],'win:{} max:{}'.format(max_win, max_mean))

        sign    = -1. if greater_is_better else 1.0
        opt.observe(rec, np.array([sign * max_mean]))
        log("params:{}".format(hyp),log_prefix)
        if verbose:
            print('\r\n{}\nIter {}, best metric: {}'.format(hyp,i, sign * opt.y.min()), flush = True)
        if done:
            break
            
    best_id   = np.argmin(opt.y.reshape(-1))
    best_hyp  = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y

    return best_hyp.to_dict(),scoresz[best_id]

def core_para(it, hyp, env = None):
    print("it:{} hyp:{}".format(it,hyp), flush = True)
    
    filename = os.path.join(base_path, '{}/Reacher'.format('Reacher_Windows'))
    
    if env is None:
        env = UnityEnvironment(file_name=filename, seed=0,no_graphics=False, worker_id=0+it)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    config = initConf_ddpg(state_size,action_size)
    
    for k in config.update_fields:
        if space.paras[k].is_numeric and space.paras[k].is_discrete:
            setattr(config, k, int(hyp[k]))
        else:
            setattr(config, k, hyp[k])

    a = config.agent_fn(config)
    scores,reward = brain(brain_name,a,config,env,it=it)
    env.close()
    return (scores,reward)

def tune_para(
    space_cfg, 
    max_iter=10, 
    hebo_cfg = None,
    greater_is_better : bool = True,
    verbose  = True,
    **kwargs
    ):
    
    sign = -1. if greater_is_better else 1.0

    if hebo_cfg is None:
        hebo_cfg = {}
    space = DesignSpace().parse(space_cfg)
    opt   = HEBO(space, rand_sample = 1, **hebo_cfg)
    
#     envs = []
#     print("UnityEnvironment Start".format(), flush = True)
    
#     f = 'Reacher_Windows'
#     no_graphics = False
#     for i in range(1,3):
#         print("UnityEnvironment i:{}".format(i), flush = True)
#         done = False
#         failed = 0
#         while not done:
#             try:
#                 envs.append(UnityEnvironment(file_name='{}/Reacher'.format(f), seed=1,no_graphics=no_graphics, worker_id=0+i))
#                 done = True
#                 print("UnityEnvironment Done i:{}".format(i), flush = True)
#                 sleep(5)
#             except Exception as e:            
#                 print("it:{} UnityEnvironment failed:{}".format(i,e), flush = True)
#                 failed += 1
#                 if failed>5:
#                     done = True
#             return            
#     print("it:{} UnityEnvironment len envs:{}".format(i,len(envs)), flush = True)
    
    scoresz = []
    
    try:
    
        for i in range(max_iter):
            rec     = opt.suggest(n_suggestions=2)
    #         hyp     = rec.iloc[0].to_dict()
    #         verbose=100, pre_dispatch='1.5*n_jobs'
            outs = Parallel(n_jobs=2,verbose=100, prefer="processes")(delayed(core_para)(i,rc.to_dict()) for i,rc in enumerate(rec.iloc))
#         envs[i],

            outs
            if verbose:
                print('\routs:{}'.format(outs), flush = True)

            a = []
            for s,r in outs:
                a.append(sign * r)

            opt.observe(rec, np.array(a))
    #         opt.observe(rec, np.array([sign * reward]))
            if verbose:
                print('\nIter %d, best metric: %g' % (i, sign * opt.y.min()), flush = True, end="")
    except Exception as e:                
#         for env in envs:
#             env.close()
#         envs = []
        raise e
#     else:
#         for env in envs:
#             env.close()
#         envs = []
    
    best_id   = np.argmin(opt.y.reshape(-1))
    best_hyp  = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y
#     if report:
#         return best_hyp.to_dict(), df_report
    return best_hyp.to_dict(),scoresz[best_id]