
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn,optim
import torch.nn.functional as F
import collections
import os
import operator

"""
DQN网络，使用全连接网络，四层，每层之间有激活函数
input_size:网络的输入向量的大小，也就是state的维度
output_size:网络的输出向量的大小，即action的数量
mem_len:经验重放的大小
"""
class DQN(nn.Module):
    def __init__(self,input_size,output_size,mem_len):
        super(DQN,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory = collections.deque(maxlen=mem_len)
        self.net = nn.Sequential(
            nn.Linear(self.input_size,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )
        #使用dueling的结构，最后的值分为价值函数和优势函数
        self.V = nn.Linear(128,1)
        self.A = nn.Linear(128,self.output_size)

    #dueling架构需要对优势值进行归一化处理：求和然后每个数减去和，得到最终的Q值
    def forward(self,input):
        net_output = self.net(input)
        v = self.V(net_output)
        #print(v.shape)
        advantage = self.A(net_output)
        #print(advantage.shape)
        advantage = advantage-torch.mean(advantage)
        q_value = v+advantage
        #print(q_value.shape)
        return q_value

    #exploration and exploitation
    def sample_action(self,inputs,epsilon):
        inputs = torch.tensor(inputs,dtype = torch.float32)
        inputs = inputs.unsqueeze(0)
        q_value  = self(inputs)
        seed = np.random.rand()
        if seed > epsilon:
            action_choice = int(torch.argmax(q_value))
        else:
            action_choice = random.choice(range(self.output_size))
        return action_choice
    
    #经验储存
    def save_trans(self,transition):
        self.memory.append(transition)
    
    #从buffer中取出batch的数据进行训练
    def sample_memory(self,batch_size):
        s_ls,a_ls,r_ls,s_next_ls = [],[],[],[]
        trans_batch = random.sample(self.memory,batch_size)
        for trans in trans_batch:
            s,a,r,s_next = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
        return torch.tensor(s_ls,dtype=torch.float32),\
            torch.tensor(a_ls,dtype=torch.int64),\
            torch.tensor(r_ls,dtype=torch.float32),\
            torch.tensor(s_next_ls,dtype=torch.float32)

#训练网络
def train_net(Q_net,Q_target,optimizer,losses,replay_time,gamma,batch_size):
    s,a,r,s_next = Q_net.sample_memory(batch_size)
    q_value = Q_net(s)
    a = torch.LongTensor(a)
    q_value = torch.gather(q_value,1,a)
    #print("q_value:{0}".format(q_value.shape))

    q_t = Q_net(s_next)
    a_index = torch.argmax(q_t,1)
    a_index = a_index.reshape((a_index.shape[0],1))
    q_target = Q_target(s_next)
    q_target = torch.gather(q_target,1,a_index)
    q_target = r+gamma*q_target #当训练结束的时候，之后就没有其余的状态了，所以直接乘上0

    #print("Q target:{0}".format(q_target.shape))

    loss = losses(q_target,q_value)
    #print("loss type:{0}".format(type(loss)))
    #print("loss size:{0}".format(loss.shape))
    #返回每一步训练的loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




#测试网络的训练
#即看输入和输出是否合法
if __name__=="__main__":
    n1 = DQN(1,5,3000)
    n2 = DQN(1,5,3000)
    n3 = DQN(1,5,3000)
            

            


