"""
随机选择节点参与CB
"""
import math
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from net import DQN,train_net
import os
from torch import optim,nn
import torch
import time

#global parameter
lambda_ = 5
global phi 
phi = sy.Symbol('phi')
global phi0
phi0 = 0.0
global n
n = 10
global global_action
global_action = []
global tmp_action 
tmp_action = []
global robots
robots = []        
global val 
val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.05)      

#reinforcement learning
learning_rate = 1e-3

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class Robot(object):
    def __init__(self,id:int=None,
                      r:int=None,
                      phi:float=None):
        self.id = id
        self.r = None
        self.phi = None
        #机器人的波束的分布
        self.alpha = None
        self.alpha_value = []
        #机器人强化学习的状态和动作
        self.state = None
        #定义机器人的神经网络
        self.model_path = "model_reward1"+str(self.id)+"\DDQN"
        self.AV_info = "info_reward1"+str(self.id)+"\Average_Reward_DDQN"
        self.input_size = n-1
        self.outputsize = 2
        self.mem_len = 30000
        self.Q_net = DQN(self.input_size,self.outputsize,self.mem_len)
        self.Q_target = DQN(self.input_size,self.outputsize,self.mem_len)
        self.Q_target.load_state_dict (self.Q_net.state_dict())
        self.optimizer = optim.Adam(self.Q_net.parameters(),lr=learning_rate)
        self.losses = nn.MSELoss()
        self.av_reward = []
        #self.reset_file()

    def reset_file(self):
        ensure_dir(self.model_path)
        ensure_dir(self.AV_info)


    #计算在向量空间的角度 
    def get_alpha(self):
        return 2*sy.pi/lambda_*self.r*(sy.cos(phi0-self.phi)-sy.cos(phi-self.phi))
    
    
    #更新自己的状态
    def new_state(self):
        size = len(val)
        sum_theta = []
        for i in range(size):
            sum_theta.append(0.0)
        for i in range(n):
            if i != self.id and global_action[i] != 0:
                for k in range(size):
                    #sum_theta[k] += abs(self.alpha_value[k]-robots[i].alpha_value[k])
                    sum_theta[k] += node_diff[self.id][i][k]
        min_phi = sum_theta.index(min(sum_theta))
        new_state = []
        for i in range(n):
            if i != self.id:
                if global_action[i] != 0:
                    new_state.append(node_diff[self.id][i][min_phi])
                else:
                    new_state.append(sy.pi.evalf(n=5))
        return new_state


    #进行新的episode学习进行机器人初始化
    #因为r=0的时候，就变成定向的天线了
    def reset(self,cur_pos):
        #self.r = np.random.randint(1,11)
        #self.phi = round(np.random.uniform(0,2*math.pi),4)
        self.r = cur_pos[0]
        self.phi = cur_pos[1]
        self.alpha = self.get_alpha()
        self.alpha_value = []
        #为了学习更加有代表性，因此进行一些归一化的操作，即将角度都规约到0到2*pi
        for i in val:
            self.alpha_value.append(((self.alpha.subs(phi,i))%(2*sy.pi)).evalf(n=5))


    #该奖励的设置的目的是从最小化 max sll出发的
    #最不想出现的情况得到的奖励是-1
    #期望的奖励是接近0
    def act(self,action):
        #old_sll = get_sll()
        tmp_action[self.id] = action
        """
        if global_action[self.id] == action:
            if max_old_action == 0:
                return -1
            else:
                return 0
        else:
            if max_new_action == 0 or max_old_action == 0:
                return 0
            else:
                new_sll = get_sll(self.id,action)
                return round(old_sll-new_sll,8)
"""
     
global old_sll
old_sll = 0.0
#将当前的分布作为参数，得到当前分布的最大SLL
def get_sll(id,action):
    num = 0
    max_val = 0.0
    f_x = 0
    f_y = 0
    #如果当前没有节点参与CB，那么得到的是0.0，否则就是大于等于0的
    for i in range(n):
        if i == id:
            if action == 1:
                num += 1
                f_x += sy.cos(robots[i].alpha)
                f_y += sy.sin(robots[i].alpha)
        else:
            if global_action[i] != 0:
                num += 1
                f_x += sy.cos(robots[i].alpha)
                f_y += sy.sin(robots[i].alpha)
    if num != 0:
        dis = sy.sqrt(f_x**2+f_y**2)/num
    else:
        dis = 0
    if dis != 0:
        left_phi = round(math.pi/30,4)
        right_phi = round(71/36*math.pi,4)
        left_pre = 1.0
        right_pre = 1.0
        left_cur = 0.0
        right_cur = 0.0
        while True:
            left_cur = (dis.subs(phi,left_phi)).evalf(n=5)
            if left_cur > left_pre:
                break
            left_pre = left_cur
            left_phi += 0.05
            if left_phi > (math.pi+0.1):
                max_val = left_pre
                break
        while True:
            right_cur = (dis.subs(phi,right_phi)).evalf(n=5)
            if right_cur > right_pre:
                break
            right_pre = right_cur
            right_phi -= 0.05
            if right_phi < (math.pi-0.1):
                max_val = max(max_val,right_pre)
                break
        #那么max SLL的范围就是从left_phi到right_phi
        #val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.05)  
        if left_phi <= right_phi:
            val_phi = np.arange(left_phi,right_phi+0.01,0.05)
            for i in val_phi:
                max_val = max(max_val,(dis.subs(phi,i)).evalf(n=5))
    return max_val

def get_old_sll():
    num = 0
    max_val = 0.0
    f_x = 0
    f_y = 0
    #如果当前没有节点参与CB，那么得到的是0.0，否则就是大于等于0的
    for i in range(n):
        if global_action[i] != 0:
            num += 1
            f_x += sy.cos(robots[i].alpha)
            f_y += sy.sin(robots[i].alpha)
    if num != 0:
        dis = sy.sqrt(f_x**2+f_y**2)/num
    else:
        dis = 0
    if dis != 0:
        left_phi = round(math.pi/30,4)
        right_phi = round(71/36*math.pi,4)
        left_pre = 1.0
        right_pre = 1.0
        left_cur = 0.0
        right_cur = 0.0
        while True:
            left_cur = (dis.subs(phi,left_phi)).evalf(n=5)
            if left_cur > left_pre:
                break
            left_pre = left_cur
            left_phi += 0.05
            if left_phi > (math.pi+0.1):
                max_val = left_pre
                break
        while True:
            right_cur = (dis.subs(phi,right_phi)).evalf(n=5)
            if right_cur > right_pre:
                break
            right_pre = right_cur
            right_phi -= 0.05
            if right_phi < (math.pi-0.1):
                max_val = max(max_val,right_pre)
                break
        #那么max SLL的范围就是从left_phi到right_phi
        #val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.05)  
        if left_phi <= right_phi:
            val_phi = np.arange(left_phi,right_phi+0.01,0.05)
            for i in val_phi:
                max_val = max(max_val,(dis.subs(phi,i)).evalf(n=5))
    return max_val
#画图
row = 3
col = 4
global fig
fig = plt.figure()
def plot_curve(av_reward,id):
    p = fig.add_subplot(row,col,id+1)
    p.grid()
    x = []
    for i in range(len(av_reward)):
        x.append(i)
    p.plot(x,av_reward,'-')
    if id% 4 == 0:
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
    plt.legend(["Robot "+str(id)])

    #plt.show()

global node_diff
node_diff = []
def ini_diff():
    size = len(val)
    for i in range(n):
        node_diff.append([])
        for j in range(n):
            node_diff[i].append([])
            if i == j:
                continue
            node_diff[i][j] = []
            for k in range(size):
                node_diff[i][j].append((sy.pi).evalf(n=5))


#计算任意两个节点的相同的Phi的取值的差值
def compute_diff():
    size = len(val)
    for i in range(n):
        for j in range(n):
            if i!=j:
                for k in range(size):
                    tmp_diff = abs((robots[i].alpha_value[k]-robots[j].alpha_value[k]).evalf(n=5))
                    node_diff[i][j][k] = tmp_diff
                    node_diff[j][i][k] = tmp_diff
                




if __name__=='__main__':
    print("test model")
    max_episode = 100

    #节点参与CB的情况
    result = []
    result_sll = []
    run_time_list = []
    for i in range(n):
        robots.append(Robot(id=i))
        #robots[i].reset_file()
        global_action.append(1)
    
    positions = np.load('../pos_10/pos_polar.npy').tolist()

    for epoch_i in range(max_episode):
        print("eposide "+str(epoch_i)+" start!")
        begin_time = time.perf_counter()
        for i in range(n):
            robots[i].reset(positions[epoch_i][i])
            #print(robots[i].alpha)
            action_tmp = np.random.randint(0,2)
            global_action[i] = action_tmp
        #此处就应该完成了随机选节点的过程
        
        if max(global_action) == 0:
            result_sll.append(1.0)
        else:
            result_sll.append(get_old_sll())
        result.append(global_action.copy())
        end_time = time.perf_counter()
        run_time = end_time-begin_time
        #print(result)
        print("episode "+str(epoch_i)+" finish!")
        run_time_list.append(run_time)
    
    result_np = np.array(result)
    result_sll_np = np.array(result_sll)
    run_time_np = np.array(run_time_list)
    np.save("random_res",result_np)
    np.save("random_sll",result_sll_np)
    np.save("random_time",run_time_list)
