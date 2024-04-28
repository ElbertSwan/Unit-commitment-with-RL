import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from rl4uc.environment import make_env
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 定义智能体
class QAgent(nn.Module):
    def __init__(self, env):
        super(QAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据硬件信息选择在cpu或gpu
        self.num_gen = env.num_gen
        self.num_nodes = 32
        self.gamma = 0.99
        self.activation = torch.tanh
        self.n_out = 2 * self.num_gen
        self.obs_size = self.process_observation(env.reset()).size
        self.mid_num_nodes = 256
        self.in_layer = nn.Linear(self.obs_size, self.num_nodes).to(self.device)
        self.mid_layer = nn.Linear(self.num_nodes, self.mid_num_nodes).to(self.device)
        self.out_layer = nn.Linear(self.mid_num_nodes, self.n_out).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-04)
        self.criterion = nn.MSELoss()
        self.loss = 0

    def process_observation(self, obs):
        """
        根据需要选择数据作为智能体接受的状态
        """
        obs_new = np.concatenate((obs['status'], [obs['timestep']]))
        return obs_new

    def forward(self, obs):
        x = torch.tensor(obs, device=self.device, dtype=torch.float)
        x = self.activation(self.in_layer(x))
        x = self.activation(self.mid_layer(x))
        return self.out_layer(x)

    def act(self, obs):
        """
        依据贪心算法选择动作
        """
        processed_obs = self.process_observation(obs)
        q_values = self.forward(processed_obs)
        q_values = q_values.reshape(self.num_gen, 2)
        action = q_values.argmax(axis=1).detach().cpu().numpy()

        return action, processed_obs

    def update(self, memory, batch_size=None):
        """
        更新网络参数
        """
        if batch_size == None:
            batch_size = memory.capacity
        data = memory.sample(batch_size)
        qs = self.forward(data['obs']).reshape(batch_size, self.num_gen, 2)
        m, n = data['act'].shape
        I, J = np.ogrid[:m, :n]
        act_cuda = torch.tensor(data['act'], device=self.device, dtype=torch.long)
        qs = qs[I, J, act_cuda]
        next_qs = self.forward(data['next_obs']).reshape(batch_size, self.num_gen, 2)
        next_acts = next_qs.argmax(axis=2).detach()
        m, n = next_acts.shape
        I, J = np.ogrid[:m, :n]
        next_qs = next_qs[I, J, next_acts]
        m, n = next_qs.shape
        rews = np.broadcast_to(data['rew'], (self.num_gen, batch_size)).T
        rews = torch.tensor(rews, device=self.device, dtype=torch.float)
        td_target = rews + self.gamma * next_qs
        criterion = nn.MSELoss()
        loss = criterion(qs, td_target)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 定义经验池
class ReplayMemory(object):

    def __init__(self, capacity, obs_size, act_dim):
        self.capacity = capacity
        self.obs_size = obs_size
        self.act_dim = act_dim
        self.act_buf = np.zeros((self.capacity, self.act_dim))
        self.obs_buf = np.zeros((self.capacity, self.obs_size))
        self.rew_buf = np.zeros(self.capacity)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_size))
        self.num_used = 0

    def store(self, obs, action, reward, next_obs):
        """Store a transition in the memory"""
        idx = self.num_used % self.capacity
        self.act_buf[idx] = action
        self.obs_buf[idx] = obs
        self.rew_buf[idx] = reward
        self.next_obs_buf[idx] = next_obs
        self.num_used += 1

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.capacity), size=batch_size, replace=False)
        data = {'act': self.act_buf[idx],
                'obs': self.obs_buf[idx],
                'rew': self.rew_buf[idx],
                'next_obs': self.next_obs_buf[idx]}
        return data

    def is_full(self):
        return self.num_used >= self.capacity

    def reset(self):
        self.num_used = 0


# 训练智能体
def train():
    MEMORY_SIZE = 200  # 经验池尺寸
    N_EPOCHS = 500  # 训练回合数
    env = make_env()  # 利用rl4uc库生成含有5个发电机的默认模型
    agent = QAgent(env)  # 初始化智能体
    memory = ReplayMemory(MEMORY_SIZE, agent.obs_size, env.num_gen)  # 初始化经验池
    print('Device is', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    log = {'mean_timesteps': [],
           'mean_reward': [],
           'loss': []
           }  # 记录数据
    for i in range(N_EPOCHS):
        if i % 10 == 0:
            print("Epoch {}".format(i))
        epoch_timesteps = []
        epoch_rewards = []
        while memory.is_full() == False:
            done = False
            obs = env.reset()
            timesteps = 0
            while not done:
                action, processed_obs = agent.act(obs)  # 智能体根据当前状态做出动作
                next_obs, reward, done = env.step(action)  # 环境根据智能体的动作发生改变
                next_obs_processed = agent.process_observation(next_obs)  # 获得最新的环境状态
                memory.store(processed_obs, action, reward, next_obs_processed)  # 存储相关数据进经验池
                obs = next_obs  # 环境状态更新
                if memory.is_full():
                    break
                timesteps += 1
                if done:
                    epoch_rewards.append(reward)
                    epoch_timesteps.append(timesteps)
        agent.update(memory)
        log['mean_timesteps'].append(np.mean(epoch_timesteps))
        log['mean_reward'].append(np.mean(epoch_rewards))
        log['loss'].append(agent.loss)
        memory.reset()
    return env, agent, log


# 训练智能体
env, agent, log = train()
# 测试智能体
done = False
obs = env.reset()
timestep = 0
while not done:
    action, processed_obs = agent.act(obs)
    next_obs, reward, done = env.step(action)
    obs = next_obs
    timestep = timestep + 1
# 绘制结果
pd.DataFrame(log['mean_reward']).plot()
plt.title('训练过程中奖励变化情况', fontdict={'weight': 'bold', 'size': 32})
plt.ylabel('奖励', fontdict={'weight': 'bold', 'size': 32})
plt.xlabel('回合', fontdict={'weight': 'bold', 'size': 32})
plt.xticks(fontproperties='Times New Roman', size=32, weight='bold')
plt.yticks(fontproperties='Times New Roman', size=32, weight='bold')
plt.legend(['奖励值'], prop={'weight': 'bold', 'size': 24})
plt.show()
pd.DataFrame(log['loss']).plot()
plt.title('训练过程中损失函数变化情况', fontdict={'weight': 'bold', 'size': 26})
plt.ylabel('损失函数值', fontdict={'weight': 'bold', 'size': 26})
plt.xlabel('回合', fontdict={'weight': 'bold', 'size': 26})
plt.xticks(fontproperties='Times New Roman', size=20, weight='bold')
plt.yticks(fontproperties='Times New Roman', size=20, weight='bold')
plt.legend(['损失函数'], prop={'weight': 'bold', 'size': 18})
plt.show()
