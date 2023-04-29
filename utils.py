import collections
import random
from env import wsn_env
import numpy as np
import torch
class ReplayBuffer:
    def __init__(self,buffer_size,use_cuda):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.cuda=use_cuda
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        if self.cuda:
            return torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).cuda(), \
               torch.tensor(r_lst).cuda(), torch.tensor(s_prime_lst, dtype=torch.float).cuda(), \
               torch.tensor(done_mask_lst).cuda()
        else:
            return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                   torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                   torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


