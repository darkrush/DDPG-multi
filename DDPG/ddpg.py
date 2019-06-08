import io
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from arguments import Singleton_arger

from model import Actor,Critic
from memory import Memory

class DDPG(object):
    def __init__(self):
        agent_args = Singleton_arger()['agent']
        self.actor_lr = agent_args['actor_lr']
        self.critic_lr = agent_args['critic_lr']
        self.lr_decay = agent_args['lr_decay']
        self.l2_critic = agent_args['l2_critic']
        self.batch_size = agent_args['batch_size']
        self.discount = agent_args['discount']
        self.tau = agent_args['tau']
        self.with_cuda = agent_args['with_cuda']
        self.buffer_size = int(agent_args['buffer_size'])
        
    def setup(self, nb_pos,nb_laser, nb_actions):
        self.lr_coef = 1
        
        model_args = Singleton_arger()['model']
        actor  = Actor (nb_pos,nb_laser, nb_actions, hidden1 = model_args['hidden1'], hidden2 = model_args['hidden2'] , layer_norm = model_args['layer_norm'])
        critic = Critic(nb_pos,nb_laser, nb_actions,hidden1 = model_args['hidden1'], hidden2 = model_args['hidden2'] , layer_norm = model_args['layer_norm'])
        self.nb_pos = nb_pos
        self.nb_laser = nb_laser
        self.actor         = copy.deepcopy(actor)
        self.actor_target  = copy.deepcopy(actor)
        self.critic        = copy.deepcopy(critic)
        self.critic_target = copy.deepcopy(critic)
        
        self.memory = Memory(self.buffer_size, (nb_actions,), (nb_pos+nb_laser,), self.with_cuda)
        
        if self.with_cuda:
            for net in (self.actor, self.actor_target, self.critic, self.critic_target):
                if net is not None:
                    net.cuda()
        
        p_groups = [{'params': [param,],
                     'weight_decay': self.l2_critic if ('weight' in name) and ('LN' not in name) else 0
                    } for name,param in self.critic.named_parameters() ]
        self.critic_optim  = Adam(params = p_groups, lr=self.critic_lr, weight_decay = self.l2_critic)
        self.actor_optim  = Adam(self.actor.parameters(), lr=self.actor_lr)
        
    def reset_noise(self):
        pass
        
    def before_epoch(self):
        pass
    
    def before_cycle(self):
        pass
        
    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        if self.with_cuda:
            s_t = s_t.cuda()
        self.memory.append(s_t, a_t, r_t, s_t1, done_t)

    def update_critic(self, batch = None, pass_batch = False):
        # Sample batch
        if batch is None:
            batch = self.memory.sample(self.batch_size)
        assert batch is not None
        tensor_obs0 = batch['obs0'].split([self.nb_pos,self.nb_laser],dim = 1)
        tensor_obs1 = batch['obs1'].split([self.nb_pos,self.nb_laser],dim = 1)
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                tensor_obs1[0], tensor_obs1[1],
                self.actor_target(tensor_obs1),
            ])
        
            target_q_batch = batch['rewards'] + self.discount*(1-batch['terminals1'])*next_q_values
        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([tensor_obs0[0],tensor_obs0[1], batch['actions']])
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()
        if pass_batch :
            return value_loss.item(), batch
        else:
            return value_loss.item()
        
    def update_actor(self, batch = None, pass_batch = False):
        if batch is None:
            batch = self.memory.sample(self.batch_size)
        assert batch is not None  
        tensor_obs0 = batch['obs0'].split([self.nb_pos,self.nb_laser],dim = 1)
        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            tensor_obs0[0],tensor_obs0[1],
            self.actor(tensor_obs0)
        ])
        
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()  
        if pass_batch :
            return policy_loss.item(), batch
        else:
            return policy_loss.item()

    def update_critic_target(self,soft_update = True):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau \
                                    if soft_update else param.data)

    def update_actor_target(self,soft_update = True):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau \
                                    if soft_update else param.data)
                                    
    def apply_lr_decay(self):
        if self.lr_decay > 0:
            self.lr_coef = self.lr_decay*self.lr_coef/(self.lr_coef+self.lr_decay)
            for (opt,base_lr) in ((self.actor_optim,self.actor_lr),(self.critic_optim,self.critic_lr)):
                for group in opt.param_groups:
                    group['lr'] = base_lr * self.lr_coef
            
    def calc_last_error(self):
        # Sample batch
        batch = self.memory.sample_last(self.batch_size)
        tensor_obs0 = batch['obs0'].split([self.nb_pos,self.nb_laser],dim = 1)
        tensor_obs1 = batch['obs1'].split([self.nb_pos,self.nb_laser],dim = 1)
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                tensor_obs1[0],tensor_obs1[1],
                self.actor_target(tensor_obs1),
            ])
            target_q_batch = batch['rewards'] + self.discount*(1-batch['terminals1'])*next_q_values
            q_batch = self.critic_target([tensor_obs0[0],tensor_obs0[1], batch['actions']])
            value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        return value_loss.item()
        
    def select_action(self, s_t, apply_noise):
        s_t = torch.tensor(np.vstack(s_t),dtype = torch.float32,requires_grad = False).cuda()
        s_t = s_t.split([self.nb_pos,self.nb_laser],dim = 1)
        
        with torch.no_grad():
            action = self.actor(s_t).cpu().numpy()
        action = np.clip(action, -1., 1.)
        return action
        
    def load_weights(self, output): 
        self.actor  = torch.load('{}/actor.pkl'.format(output) )
        self.critic = torch.load('{}/critic.pkl'.format(output))
            
    def save_model(self, output):
        torch.save(self.actor ,'{}/actor.pkl'.format(output) )
        torch.save(self.critic,'{}/critic.pkl'.format(output))
            
    def get_actor_buffer(self):
        actor_buffer = io.BytesIO()
        torch.save(self.actor, actor_buffer)
        return actor_buffer