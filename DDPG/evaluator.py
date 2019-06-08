import numpy as np
import torch
import argparse
import gym.spaces
import gym
import time
import os
import pickle

from MPE.make_env import make_env

from multiprocessing import Process, Queue

class Evaluator(object):
    def __init__(self):
        self.env = None
        self.actor = None

        self.visualize = False
        
    def setup(self, env_name, logger,  num_episodes = 10,  model_dir = None,
              multi_process = True, visualize = False, rand_seed = -1):
        self.env_name = env_name
        self.logger = logger
        self.num_episodes = num_episodes
        self.model_dir = model_dir
        self.multi_process = multi_process
        self.visualize = visualize
        self.rand_seed = rand_seed
        if self.multi_process :
            self.queue = Queue(maxsize = 1)
            self.sub_process = Process(target = self.start_eval_process,args = (self.queue,))
            self.sub_process.start()
        else :
            self.setup_gym_env()
            
    def setup_gym_env(self):
        self.env = make_env(self.env_name,timelimit = 300)
        if self.rand_seed >= 0:
            self.env.seed(self.rand_seed)
        self.action_scale = (self.env.action_space[0].high - self.env.action_space[0].low)/2.0
        self.action_bias = (self.env.action_space[0].high + self.env.action_space[0].low)/2.0
  
    def start_eval_process(self,queue):
        self.setup_gym_env()
        while True:
            item = queue.get(block = True)
            if item is '__close__':
                break
            if isinstance(item,str):
                self.laod_from_file(item)
            else:
                self.run_eval(item)
            
    def load_from_buffer(self, buffer):
        self.actor = torch.load(buffer)
        
    def laod_from_file(self,model_dir = None):
        if model_dir is None:
            model_dir = self.model_dir
        assert model_dir is not None
        self.actor = torch.load(os.path.join(model_dir,'actor.pkl'))
        
    def run_eval(self,total_cycle):
        assert self.actor is not None
        observation = None
        result = []
        for episode in range(self.num_episodes):

            observation,info = self.env.reset()
            episode_steps = 0
            episode_reward = 0.
            assert observation is not None
            all_done = False
            timedone = False
            while not (all_done or timedone):

                obs = torch.tensor(np.vstack(observation),dtype = torch.float32,requires_grad = False).cuda()
                obs = obs.split([5,32],dim = 1)

                with torch.no_grad():
                    action = self.actor(obs).cpu().numpy()
                action = np.clip(action, -1., 1.)
                action = action * self.action_scale + self.action_bias
                observation, reward, done, info = self.env.step(action)
                all_done = True
                for done_ in done:
                    all_done = all_done and done_
                timedone = info['TimeLimit.truncated']
                if self.visualize & (episode == 0):
                    self.env.render(mode='human',close=False)
                for re in reward:
                    episode_reward += re
                episode_steps += 1
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        result_mean = result.mean()
        result_std = result.std(ddof = 1)
        if self.logger is not None :
            self.logger.trigger_log( 'eval_reward_mean',result_mean, total_cycle)
            self.logger.trigger_log( 'eval_reward_std',result_std, total_cycle)
        localtime = time.asctime( time.localtime(time.time()) )
        print("{} eval : cycle {:<5d}\treward mean {:.2f}\treward std {:.2f}".format(localtime,total_cycle,result_mean,result_std))
        
    def trigger_load_from_file(self, actor_dir):
        if self.multi_process :
            self.queue.put(actor_dir,block = True)
        else:
            self.laod_from_file(actor_dir)
    
    def trigger_eval_process(self,total_cycle):
        if self.multi_process :
            self.queue.put(total_cycle,block = True)
        else :
            self.run_eval(total_cycle)

    def trigger_close(self):
        if self.multi_process :
            self.queue.put('__close__',block = True)

    def __del__(self):
        if self.env is not None:
            self.env.close()
    
    
Singleton_evaluator = Evaluator()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Eval DDPG')
    parser.add_argument('--logdir', default=None, type=str, help='result output dir')
    parser.add_argument('--env', default=None, type=str, help='open-ai gym environment')
    parser.add_argument('--model-dir', default=None, type=str, help='actor for evaluation')
    parser.add_argument('--num-episodes', default=10, type=int, help='number of episodes')
    parser.add_argument('--visualize', dest='visualize', action='store_true',help='enable render in evaluation progress')
    parser.set_defaults(visualize=False)
    
    args = parser.parse_args()
    if args.logdir is not None:
        with open(args.logdir,'rb') as f:
            exp_args = pickle.load(f)
            args.env = exp_args.env
            args.model_dir = exp_args.result_dir
            
    assert args.env is not None
    assert args.model_dir is not None
    
    Singleton_evaluator.setup(env_name = args.env,
                              logger = None,
                              num_episodes = 10,
                              model_dir = args.model_dir,
                              multi_process = False,
                              visualize = args.visualize,
                              rand_seed = 0)
                              
    Singleton_evaluator.laod_from_file()
    Singleton_evaluator.run_eval(0)
