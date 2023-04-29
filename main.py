from env import wsn_env
import os
from HGFF import HGFF
from utils import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import copy
import argparse

learning_rate = 0.0001
gamma = 0.98
buffer_limit = 5000
batch_size = 64
epsilon = 0.99
eps_start=0.99
eps_end=0.01
eps_anneal=5e-5
seed=1234
use_cuda = torch.cuda.is_available()
UPDATE_TARGET_INTERVAL=200
PRINT_INTERVAL = 50
mean_rewards=[]

def train(double_q, q, q_target, memory, optimizer):
    for i in range(4):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        s_prime_1=copy.deepcopy(s_prime)
        q_out = q(s)
        q_a = q_out.gather(1, a)

        if double_q:
            greedy_actions=q(s_prime).argmax(1,True)
            max_q_prime = q_target(s_prime_1).gather(1,greedy_actions)
        else:
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", default=1.1e5, type=int)
    parser.add_argument("--double_DQN", default=True, type=bool)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dyn_map", default=True, type=bool)
    parser.add_argument("--map_index", default=0, type=int)
    parser.add_argument("--map_type", default=0, type=int)
    config = parser.parse_args()
    print("start from map:", config.map_type, "-", config.map_index," ,dyn:", config.dyn_map, ", seed:",seed)
    run(config)

def set_global_seed(seed, env):
    torch.manual_seed(seed)
    env.set_seed(seed)
    np.random.seed(seed)

def run(config):

    DOUBLE=True
    n_steps=0
    sum_rewards = 0
    global epsilon
    start=time.time()

    env = wsn_env(dyn=config.dyn_map,map_type=config.map_type,map_name=config.map_index)

   
    env.reset()
    if seed:
        set_global_seed(seed,env)
    a_size = env.get_a_size()
    # =========networks==========
    q = HGFF(n_obs_in=5,n_action=a_size,n_layers=3,n_features=64,n_hid_readout=[],tied_weights=False)
    q_target = HGFF(n_obs_in=5, n_action=a_size, n_layers=3, n_features=64, n_hid_readout=[], tied_weights=False)
    q_target.load_state_dict(q.state_dict())


    if use_cuda:
        q=q.to(device="cuda")
        q_target=q_target.to(device="cuda")

    memory = ReplayBuffer(buffer_limit,use_cuda)

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    route = [[]]

    for N_EPI in range(int(config.train_episodes)):
        s = env.reset()
        done = False
        temp_route = []
        while not done:
            if use_cuda:
                s = torch.tensor(s).cuda()
                s=s.to(torch.float32)
            else:
                s = np.array(s)
                s=torch.from_numpy(s).float()

            a = q.sample_action(s, epsilon,a_size)
            s_prime, r, done = env.step(a)
            temp_route.append(a)
            n_steps += 1
            done_mask = 0.0 if done else 1.0
            memory.put((s.cpu().numpy().tolist(), a, r, s_prime, done_mask))
            s = s_prime
            sum_rewards += r

            if done:
                epsilon = epsilon - eps_anneal if epsilon > eps_end else epsilon
                break

        if memory.size() > 5000:
            train(DOUBLE, q, q_target, memory, optimizer)
            if N_EPI % UPDATE_TARGET_INTERVAL ==0:
                q_target.load_state_dict(q.state_dict())

        if N_EPI % PRINT_INTERVAL == 0 and N_EPI != 0:
            num=int(N_EPI / PRINT_INTERVAL)
            log_time=time.time()
            avg_reward=sum_rewards / PRINT_INTERVAL
            mean_rewards.append(avg_reward)
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}, time: {:.1f}".format(
                N_EPI, avg_reward, memory.size(), epsilon * 100, (log_time-start)/num))
     


if __name__ == '__main__':
    main()