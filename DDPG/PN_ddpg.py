import copy
import numpy as np
import torch
from ddpg import DDPG
from arguments import Singleton_arger
from noise import AdaptiveParamNoiseSpec

class parameter_noise_DDPG(DDPG):
    def __init__(self):
        super(parameter_noise_DDPG, self).__init__()
        exploration_args = Singleton_arger()['exploration']
        self.parameter_noise = AdaptiveParamNoiseSpec( initial_stddev=exploration_args['stddev'], desired_action_stddev=exploration_args['stddev'], adoption_coefficient=1.01)

        
    def setup(self,nb_states, nb_actions):
        super(parameter_noise_DDPG, self).setup(nb_states, nb_actions)
        self.rollout_actor   = copy.deepcopy(self.actor)
        self.measure_actor = copy.deepcopy(self.actor)
        
        if self.with_cuda:
            for net in (self.rollout_actor, self.measure_actor):
                if net is not None:
                    net.cuda()

                        
    def reset_noise(self):
        for target_param, param in zip(self.rollout_actor.parameters(), self.actor.named_parameters()):
            name, param = param
            if 'LN' not in name:
                target_param.data.copy_(param.data + torch.normal(mean=torch.zeros_like(param),std=torch.full_like(param,self.parameter_noise.current_stddev)))
            else:
                target_param.data.copy_(param.data)


    def before_cycle(self):
        self.adapt_param_noise()
                
    def adapt_param_noise(self):
        for target_param, param in zip(self.measure_actor.parameters(), self.actor.named_parameters()):
            name, param = param
            if 'LN' not in name:
                target_param.data.copy_(param.data + torch.normal(mean=torch.zeros_like(param),std=torch.full_like(param,self.parameter_noise.current_stddev)))
            else:
                target_param.data.copy_(param.data)
        batch = self.memory.sample(self.batch_size)
        tensor_obs0 = batch['obs0']
        with torch.no_grad():
            distance = torch.mean(torch.sqrt(torch.sum((self.actor(tensor_obs0)- self.measure_actor(tensor_obs0))**2,1)))
        self.parameter_noise.adapt(distance)
    
    def select_action(self, s_t, apply_noise):
        s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        if self.with_cuda:
            s_t = s_t.cuda()
        with torch.no_grad():
            if apply_noise:
                action = self.rollout_actor(s_t).cpu().numpy().squeeze(0)
            else:            
                action = self.actor(s_t).cpu().numpy().squeeze(0)
        action = np.clip(action, -1., 1.)
        return action