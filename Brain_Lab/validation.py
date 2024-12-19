import numpy as np
import random
import torch
import time
from torch import nn
import gym
import cv2
import copy
import matplotlib.pyplot as plt

#human 
env = gym.make('BreakoutDeterministic-v4', render_mode='human')



obs = env.reset()
obs = obs[0]  

#game length
# max_steps = 500
N_FRAMES = 4
n_episodes = 3000
max_steps = 1000
er_capacity = 50000  # yo ali badi vayo ra? kina ki 300k hunxa max experience, tetro badi ni save garnu pardaina hola 
n_acts = env.action_space.n # 0: no-op, 1: start game, 2: right, 3: left
train_batch_size = 16
learning_rate = 2.5e-4
target_update_delay = 200
print_freq = 10
update_freq = 2
frame_skip = 4
n_anneal_steps = 10000 # Anneal over 1m steps in paper
epsilon = lambda step: np.clip(1 - 0.9 * (step/n_anneal_steps), 0.1, 1) # Anneal over 1m steps in paper, 100k here



# print(env.step(0))


def filter_obs(obs, resize_shape=(84, 110), crop_shape=None):
    assert(type(obs) == np.ndarray), "The observation must be a numpy array!"
    assert(len(obs.shape) == 3), "The observation must be a 3D array!"

    obs = cv2.resize(obs, resize_shape, interpolation=cv2.INTER_LINEAR)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs / 255.
    if crop_shape:
        crop_x_margin = (resize_shape[1] - crop_shape[1]) // 2
        crop_y_margin = (resize_shape[0] - crop_shape[0]) // 2
        
        x_start, x_end = crop_x_margin, resize_shape[1] - crop_x_margin
        y_start, y_end = crop_y_margin, resize_shape[0] - crop_y_margin
        
        obs = obs[x_start:x_end, y_start:y_end]
    
    return obs

def get_stacked_obs(obs, prev_frames):
    if not prev_frames:
        prev_frames = [obs] * (N_FRAMES - 1)
        
    prev_frames.append(obs)    # po1, po2, po3, ob (4,110,84)
    stacked_frames = np.stack(prev_frames)  #(4,110,84)
    prev_frames = prev_frames[-(N_FRAMES-1):] # (3,100,84)
    
    return stacked_frames, prev_frames  #(4,110,84)  ,  (3,100,84)


def format_reward(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    return 0

def preprocess_obs(obs, prev_frames):
    filtered_obs = filter_obs(obs)   #resize, rgb, normalize   (110,84)
    stacked_obs, prev_frames = get_stacked_obs(filtered_obs, prev_frames)
    return stacked_obs, prev_frames

class DQN(nn.Module):
    def __init__(self, n_acts):
        super(DQN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(N_FRAMES, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(32 * 12 * 9, 256),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(256, n_acts))
        
    def forward(self, obs):
        q_values = self.layer1(obs)
        q_values = self.layer2(q_values)
        
        # 2015 model: (32, 8x8, 4), (64, 4x4, 2), (64, 3x3, 1), (512)
        q_values = q_values.view(-1, 32 * 12 * 9)
        q_values = self.layer3(q_values)
        q_values = self.layer4(q_values)     #layer 4 bata ta 4 ota output aauxa, yo 4 ota output kun type ko hunxa? array? or what?
        
        return q_values
    
    def train_on_batch(self, target_model, optimizer, obs, acts, rewards, next_obs, terminals, gamma=0.99):
        next_q_values = self.forward(next_obs)
        max_next_acts = torch.max(next_q_values, dim=1)[1].detach()
        
        target_next_q_values = target_model.forward(next_obs)
        max_next_q_values = target_next_q_values.gather(index=max_next_acts.view(-1, 1), dim=1)
        max_next_q_values = max_next_q_values.view(-1).detach()


        terminal_mods = 1 - terminals
        actual_qs = rewards + terminal_mods * gamma * max_next_q_values
            
        pred_qs = self.forward(obs)
        pred_qs = pred_qs.gather(index=acts.view(-1, 1), dim=1).view(-1)
        
        loss = torch.mean((actual_qs - pred_qs) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



class ExperienceReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []    #yo list of list hunxa, [[obs,act,rew,next_obs,done],[obs,act,rew,next_obs,done],] 
        
    def add_step(self, step_data):
        self.data.append(step_data)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]  #badi vaye, first ko nikalne.
            
    def sample(self, n):
        n = min(n, len(self.data))
        indices = np.random.choice(range(len(self.data)), n, replace=False)
        samples = np.asarray(self.data)[indices]
        
        state_data = torch.tensor(np.stack(samples[:, 0])).float().cuda()
        act_data = torch.tensor(np.stack(samples[:, 1])).long().cuda()
        reward_data = torch.tensor(np.stack(samples[:, 2])).float().cuda()
        next_state_data = torch.tensor(np.stack(samples[:, 3])).float().cuda()
        terminal_data = torch.tensor(np.stack(samples[:, 4])).float().cuda()
        
        return state_data, act_data, reward_data, next_state_data, terminal_data




# er = ExperienceReplay(er_capacity)
# model = DQN(n_acts=env.action_space.n).cuda()
# target_model = copy.deepcopy(model)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-6)
# all_rewards = []
# global_step = 0
# save_episodes = {500, 1000, 1500, 2000,2500,3000}


# for episode in range(n_episodes):
#     prev_frames = []
#     obs, prev_frames = preprocess_obs(env.reset()[0], prev_frames) #input: (H,W,3)   output: #(4,110,84)  ,  (3,100,84)

    
#     episode_reward = 0
#     step = 0

#     while step < max_steps:

#         ### Enact a step ###
        
#         if np.random.rand() < epsilon(global_step):
#             act = np.random.choice(range(n_acts))
#         else:
#             obs_tensor = torch.tensor(obs).unsqueeze(0).float().cuda() # (4,110,80) -> tensor of (1,4,100,80) i.e batch dimension introduce gareko
#             q_values = model(obs_tensor)[0]  #input: (1,4,110,80)  #output (4)
#             q_values = q_values.cpu().detach().numpy() #numpy ma convert
#             act = np.argmax(q_values) #jun ko max q values, tei leko
        
#         #kun action line vanne aaune ta ho yo.  yo model le predicted gareko jun ma q values badi xa tyo aaune
#         #hai? yes, aba yo model chain, time anusar ramro hunxa, so we act always best? 
#         #yes first, first ma ramro action predict gardaina so we do backprop


#         #ati samma kun action line vanne matra xa hai 


#         cumulative_reward = 0
#         #ata 4 frames ko reward jodeko hoina? 
#         for _ in range(frame_skip):
#             next_obs, reward, done, _,_ = env.step(act)
#             cumulative_reward += reward
#             if done or step >= max_steps:
#                 break


#         episode_reward += cumulative_reward
#         reward = format_reward(cumulative_reward)

#         next_obs, prev_frames = preprocess_obs(next_obs, prev_frames)
#         er.add_step([obs, act, reward, next_obs, int(done)])   #euta experience save gareko? -> yes
#         obs = next_obs
        
#         ## Train on a minibatch ###

#         if global_step % update_freq == 0:
#             obs_data, act_data, reward_data, next_obs_data, terminal_data = er.sample(train_batch_size)  #ok batch_size jati ko data line, now train garne.
#             model.train_on_batch(target_model,optimizer, obs_data, act_data, reward_data, next_obs_data, terminal_data)  #ok train vayo, now parameters are tweaks to predict correct q values.
        
#         if global_step and global_step % target_update_delay == 0:
#             target_model = copy.deepcopy(model)

#         step += 1
#         global_step += 1
        
#         if done:
#             break
            
#     all_rewards.append(episode_reward)
    
#     if episode % print_freq == 0:
#         print('Episode #{} | Step #{} | Epsilon {:.2f} | Avg. Reward {:.2f}'.format(
#             episode, global_step, epsilon(global_step), np.mean(all_rewards[-print_freq:])))


#     if episode in save_episodes:
#         torch.save(model.state_dict(), f'dqn_breakout_episode_{episode}.pt')
#         print(f'Model saved at episode {episode}')


model = DQN(n_acts=env.action_space.n).cuda()
model.load_state_dict(torch.load('dqn_breakout_episode_1500.pt'))
model.eval()


prev_frames = []
obs, prev_frames = preprocess_obs(env.reset()[0], prev_frames)

for step in range(max_steps):
    if np.random.rand() < 0.05:
        act = np.random.choice(range(n_acts))
    else:
        obs_tensor = torch.tensor([obs]).float().cuda()
        q_values = model(obs_tensor)[0]
        q_values = q_values.cpu().detach().numpy()
        act = np.argmax(q_values)

    for _ in range(frame_skip):
        next_obs, reward, done, _,_ = env.step(act)
        if done or step >= max_steps:
            break
            
        env.render()
        time.sleep(0.05)
        
    if done:
        break

    obs, prev_frames = preprocess_obs(next_obs, prev_frames)



# torch.save(model, 'dqn_breakout_r92.pt')
