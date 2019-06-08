import argparse
import numpy
import torch
import os
import pickle

class Args(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='DDPG on pytorch')
        #Exp & Dir name 
        parser.add_argument('--output', default='results/', type=str, help='result output dir')
        parser.add_argument('--env', default='MAC_4', type=str, help='open-ai gym environment')
        parser.add_argument('--exp-name', default='0', type=str, help='exp dir name')
        parser.add_argument('--result-dir',default=None, type=str, help='whole result dir name')
        
        #Training args
        parser.add_argument('--nb-epoch', default=100, type=int, help='number of epochs')
        parser.add_argument('--nb-cycles-per-epoch', default=100, type=int, help='number of cycles per epoch')
        parser.add_argument('--nb-rollout-steps', default=100, type=int, help='number rollout steps')
        parser.add_argument('--nb-train-steps', default=100, type=int, help='number train steps')
        #parser.add_argument('--max-episode-length', default=1000, type=int, help='max steps in one episode')
        parser.add_argument('--nb-warmup-steps', default=100, type=int, help='time without training but only filling the replay memory')
        parser.add_argument('--train-mode', default=0, type=int, help='traing mode')
        
        #Model args
        parser.add_argument('--hidden1', default=64, type=int, help='number of hidden1')
        parser.add_argument('--hidden2', default=64, type=int, help='number of hidden2')
        parser.add_argument('--not-LN', dest='layer_norm', action='store_false',help='model without LayerNorm')
        parser.set_defaults(layer_norm=True)
        
        #DDPG args
        parser.add_argument('--actor-lr', default=0.0001, type=float, help='actor net learning rate')
        parser.add_argument('--critic-lr', default=0.001, type=float, help='critic net learning rate')
        parser.add_argument('--lr-decay', default=0, type=float, help='critic lr decay')
        parser.add_argument('--l2-critic', default=0.01, type=float, help='critic l2 regularization')
        parser.add_argument('--batch-size', default=128, type=int, help='minibatch size')
        parser.add_argument('--discount', default=0.95, type=float, help='reward discout')
        parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
        parser.add_argument('--nocuda', dest='with_cuda', action='store_false',help='disable cuda')
        parser.set_defaults(with_cuda=True)
        parser.add_argument('--buffer-size', default=1e6, type=int, help='memory buffer size')
        #Exploration args
        parser.add_argument('--action-noise', dest='action_noise', action='store_true',help='enable action space noise')
        parser.set_defaults(action_noise=False)
        parser.add_argument('--parameter-noise', dest='parameter_noise', action='store_true',help='enable parameter space noise')
        parser.set_defaults(parameter_noise=False)
        parser.add_argument('--stddev', default=0.6, type=float, help='action noise stddev')
        parser.add_argument('--noise-decay', default=0, type=float, help='action noise decay')
        parser.add_argument('--SGLD-mode', default=0, type=int, help='SGLD mode, 0: no SGLD, 1: actor sgld only, 2: critic sgld only, 3: both actor & critic')
        parser.add_argument('--no-SGLD-noise', dest='SGLD_noise', action='store_false',help='disable SGLD noise')
        parser.set_defaults(SGLD_noise=True)
        parser.add_argument('--num-pseudo-batches', default=0, type=int, help='SGLD pseude batch number')
        parser.add_argument('--nb-rollout-update', default=50, type=int, help='number of SGLD rollout actor step')
        parser.add_argument('--temp', default=1, type=float, help='Temperature of SGLD')
        #Other args
        parser.add_argument('--rand-seed', default=314, type=int, help='random_seed')
        
        
        parser.add_argument('--mp', dest='multi_process', action='store_true',help='enable multi process')
        parser.set_defaults(multi_process=False)
        
        args = parser.parse_args()
        if args.result_dir is None:
            args.result_dir = os.path.join(args.output, args.env, args.exp_name, "{}".format(args.rand_seed))
            
        os.makedirs(args.result_dir, exist_ok=False)
        
        if not torch.cuda.is_available():
            args.with_cuda = False
        with open(os.path.join(args.result_dir,'args.pkl'),'wb') as f:
            pickle.dump(args, file = f)  
        with open(os.path.join(args.result_dir,'args.txt'),'w') as f:
            print(args,file = f)
        
        if args.rand_seed >= 0:
            if args.with_cuda:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed_all(args.rand_seed)
            torch.manual_seed(args.rand_seed)
            numpy.random.seed(args.rand_seed)
        assert  args.action_noise + args.parameter_noise + (args.SGLD_mode is not 0) <= 1
        args_main  = { key :  args.__dict__[key] for key in ('output','env', 'exp_name', 'result_dir','multi_process','rand_seed')}
        args_model = { key :  args.__dict__[key] for key in ('hidden1', 'hidden2', 'layer_norm')}
        args_train = { key :  args.__dict__[key] for key in ('nb_epoch', 'nb_cycles_per_epoch', 'nb_rollout_steps', 'nb_train_steps', 'nb_warmup_steps', 'train_mode')}
        args_exploration = {key : args.__dict__[key] for key in ('action_noise','parameter_noise','stddev','noise_decay','SGLD_mode','SGLD_noise','num_pseudo_batches','nb_rollout_update','temp')}
        args_agent = { key :  args.__dict__[key] for key in ('actor_lr','critic_lr','lr_decay','l2_critic','batch_size','discount','tau','buffer_size','with_cuda')}
        
        self.args_dict={'main':args_main, 'model':args_model, 'train':args_train, 'exploration':args_exploration, 'agent':args_agent}
    def __call__(self):
        return self.args_dict
Singleton_arger = Args()
