import copy
import numpy as np
import torch
from ddpg import DDPG
from noise import OrnsteinUhlenbeckActionNoise
from arguments import Singleton_arger

class action_noise_DDPG(DDPG):
    def __init__(self):
        super(action_noise_DDPG, self).__init__()

    def setup(self, nb_pos,nb_laser, nb_actions):
        super(action_noise_DDPG, self).setup(nb_pos,nb_laser, nb_actions)
        self.nb_pos = nb_pos
        self.nb_laser = nb_laser   
        exploration_args = Singleton_arger()['exploration']
        self.noise_decay = exploration_args['noise_decay']
        self.noise_coef = 1
        self.rollout_actor = copy.deepcopy(self.actor)
        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(exploration_args['stddev']) * np.ones(nb_actions))
        if self.with_cuda:
            for net in (self.rollout_actor,):
                if net is not None:
                    net.cuda()
  
    def reset_noise(self):
        self.action_noise.reset()
    
    def before_epoch(self):
        self.apply_noise_decay()
        
    def apply_noise_decay(self):
        if self.noise_decay > 0:
            self.noise_coef = self.noise_decay*self.noise_coef/(self.noise_coef+self.noise_decay)

    def select_action(self, s_t, apply_noise):
        s_t = torch.tensor(np.vstack(s_t),dtype = torch.float32,requires_grad = False).cuda()
        s_t = s_t.split([self.nb_pos,self.nb_laser],dim = 1)
        #s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        #if self.with_cuda:
        #    s_t = s_t.cuda()
        with torch.no_grad():
            action = self.actor(s_t).cpu().numpy()
        if apply_noise:
            action += max(self.noise_coef, 0)*self.action_noise()
        action = np.clip(action, -1., 1.)
        return action