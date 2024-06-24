import math
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from net import DQN,train_net
import os
from torch import optim,nn
import torch
import logging
from datetime import datetime
import time

#global parameter
#frequency = 2.4GHz, then wavelength=0.125m
global lambda_
lambda_=0.125
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
#取值有点小了，可以取0.1的间隔
#global val 
#val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.05)    
global val 
val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.1)

#new parameters
global ini_energy
ini_energy = 3*((10)**5)
global l_data
l_data = 2*((10)**6)
global e_cct
e_cct = 10**(-7)
global e_tx
e_tx = 10**(-12)
global path_loss
path_loss = 3
global dis_controller
dis_controller = 10**3
global omega_
omega_ = 0.5
global velocity
velocity = np.load('./velocity.npy',allow_pickle=True).tolist()
global motion_time
motion_time = np.load('./time.npy',allow_pickle=True).tolist()
global positions
positions = np.load('./pos_polar.npy').tolist()
global k1
k1 = 7.4
global k2
k2 = 0.29

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
        self.energy = None
        #机器人的波束的分布
        self.alpha = None
        self.alpha_value = []
        #机器人强化学习的状态和动作
        self.state = None
        #定义机器人的神经网络
        self.model_path = "../train_res/model_"+str(self.id)+"/DDQN"
        self.AV_info = "../train_res/info_"+str(self.id)+"/"
        #包括位置状态和能量状态,2*(n-1)
        self.input_size = 2*(n-1)
        self.outputsize = 2
        self.mem_len = 30000
        self.Q_net = DQN(self.input_size,self.outputsize,self.mem_len)
        self.Q_target = DQN(self.input_size,self.outputsize,self.mem_len)
        self.Q_target.load_state_dict (self.Q_net.state_dict())
        self.optimizer = optim.Adam(self.Q_net.parameters(),lr=learning_rate)
        self.losses = nn.MSELoss()
        self.av_reward = []
        self.ep_reward = []     #记录每个episode的节点reward
        self.reset_file()

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
        #state for AF
        for i in range(n):
            if i != self.id:
                if global_action[i] != 0:
                    new_state.append(node_diff[self.id][i][min_phi])
                else:
                    new_state.append(sy.pi.evalf(n=5))
        #state for energy
        for i in range(n):
            if i != self.id:
                new_state.append(round(robots[i].energy-self.energy,5))
        
        return new_state


    #机器人随机运动，所以每次随机初始化应该就可以了
    def ini_robot(self):
        #考虑的范围是100m*100m的范围
        self.r = 0
        self.phi = 0
        self.energy = ini_energy

    def reset(self,index,id):
        #考虑的范围是100m*100m的范围
        self.r = positions[index][id][0]
        self.phi = positions[index][id][1]
        self.alpha = self.get_alpha()
        self.alpha_value = []
        #为了学习更加有代表性，因此进行一些归一化的操作，即将角度都规约到0到2*pi
        for i in val:
            self.alpha_value.append(((self.alpha.subs(phi,i))%(2*sy.pi)).evalf(n=5))
        e_motion = round(motion_time[index]*(k1*velocity[index][id]+k2),5)
        self.energy -= e_motion

    #采取动作
    def act(self,action):
        tmp_action[self.id] = action
        #global_action[self.id] = action

#将当前的分布作为参数，得到当前分布的最大SLL
def get_sll():
    num = 0
    max_val = 0.0
    f_x = 0
    f_y = 0
    density = 0.1
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
            left_phi += density
            if left_phi > (math.pi+0.1):
                max_val = left_pre
                break
        while True:
            right_cur = (dis.subs(phi,right_phi)).evalf(n=5)
            if right_cur > right_pre:
                break
            right_pre = right_cur
            right_phi -= density
            if right_phi < (math.pi-0.1):
                max_val = max(max_val,right_pre)
                break
        #那么max SLL的范围就是从left_phi到right_phi
        #val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.05)  
        if left_phi <= right_phi:
            val_phi = np.arange(left_phi,right_phi+0.01,0.1)
            for i in val_phi:
                max_val = max(max_val,(dis.subs(phi,i)).evalf(n=5))
    return max_val

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
              
def robo_info():
    logger.info("robot information")
    logger.info("#"*10)
    for i in range(n):
        logger.info("robot:{0},r:{0},phi:{1},energy:{2}".format(i,robots[i].r,robots[i].phi,robots[i].energy))
        logger.info("state:{0}".format(robots[i].state))
    logger.info("#"*10)


def update_energy1():
    cur_num = sum(global_action)
    delta_e = 0.0
    if cur_num != 0:
        delta_e = round(l_data*(e_cct+e_tx*((dis_controller)**path_loss)/(cur_num**2)),5)
    for k in range(n):
        if global_action[k] == 1:
            robots[k].energy -= delta_e

def update_energy2(best):
    cur_num = sum(best)
    delta_e = 0.0
    if cur_num != 0:
        delta_e = round(l_data*(e_cct+e_tx*((dis_controller)**path_loss)/(cur_num**2)),5)
    for k in range(n):
        if best[k] == 1:
            robots[k].energy -= delta_e
            robots[k].energy = round(robots[k].energy,5)

def nor_state(state):
    max_ = max(state)
    min_ = min(state)
    for i in range(len(state)):
        state[i] = (state[i]-min_)/(max_-min_)
    return state


global f_x
f_x = []
global f_y
f_y = []
     

def get_f():
    for i in range(len(val)):
        f_x[i] = 0.0
        f_y[i] = 0.0
        #sum_alpha[i] = 0
        for j in range(n):
            if global_action[j] == 1:
                    f_x[i] += sy.cos(robots[j].alpha_value[i])
                    f_y[i] += sy.sin(robots[j].alpha_value[i])
        #sum_alpha[i] = round(float(sy.sqrt(fx[i]**2+fy[i]**2)),5)
    

#计算所有的alpha的和，其实就是AF的值
def get_sll():
    fx = f_x.copy()
    fy = f_y.copy()
    num = sum(global_action)
    f = []
    for i in range(len(val)):
        f.append(sy.sqrt(fx[i]**2+fy[i]**2))
        f[i] = round(float(f[i]/num),5)
    #进行遍历，找到MSLL
    left = 0
    right = len(f)-1
    i = 0
    while i<right:
        if f[i]<f[i+1]:
            left = i
            break
        else:
            i += 1
    i = len(f)-1
    while i>0:
        if f[i-1]>f[i]:
            right = i
            break
        else:
            i -= 1
    max_val = 0.0
    if left < right:
        for i in range(left,right+1):
            max_val = max(max_val,f[i])
    #print('left:{0},right:{1}'.format(left,right))
    if max_val == 0:
        return 1
    else:
        return max_val

if __name__=='__main__':
    logger = logging.getLogger(__name__)
    current_date_time = datetime.now()
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    fh = logging.FileHandler(f"./test/model_test.txt")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.basicConfig(level=logging.NOTSET)

    episode_model_param= 800
    max_episode = 100
    max_step = 200
    epsilon = 0.12535073493989754

    #结果记录
    res_sll = []
    res = []
    #cb前后的机器人能量变化
    res_afcb_energy = []
    res_becb_energy = []
    res_f2_nor = []
    res_f2 = []
    res_f1 = []
    res_f = []
    best_f = 20.0
    num_all = []
    time_all = []

    for i in range(n):
        robots.append(Robot(id=i))
        robots[i].reset_file()
        global_action.append(1)
        tmp_action.append(1)
    ini_diff()
    #tmp_action = global_action
    for i in range(len(val)):
        f_x.append(0.0)
        f_y.append(0.0)
    
    for i in range(n):
        robots[i].Q_net = torch.load(robots[i].model_path+str(episode_model_param)+".pth")
        robots[i].Q_target.load_state_dict(robots[i].Q_net.state_dict())
        robots[i].ini_robot()
        print("load model")

    for epoch_i in range(max_episode):
        #print("eposide "+str(epoch_i)+" start!")
        logger.info("eposide "+str(epoch_i)+" start!")
        best_f = 20.0
        #记录机器人的初始能量
        ini_energy = []
        for i in range(n):
            robots[i].reset(epoch_i,i)
            global_action[i] = 1
            tmp_action[i] = 1
            ini_energy.append(robots[i].energy)
        #ini_diff()
        compute_diff()
        for i in range(n):
            robots[i].state = robots[i].new_state()
        
        #robo_info()
        start_time = time.perf_counter()
        for j in range(max_step):
            #print("step "+str(j)+" start!")
            #logger.info("step "+str(j)+" start!")
            for k in range(n):
                a0 = robots[k].Q_net.sample_action(robots[k].state)
                robots[k].act(a0)

            for k in range(n):
                global_action[k] = tmp_action[k]
            #print(global_action)
            update_energy1()
            for k in range(n):
                robots[k].state = robots[k].new_state()

            get_f()
            f1 = get_sll()
            f2 = 1.0
            tmp_state = []
            num = sum(global_action)
            delta_e = 0.0
            if num != 0:
                delta_e = l_data*(e_cct+e_tx*((dis_controller)**path_loss)/(num**2))
            for k in range(n):
                if global_action[k] == 1:
                    tmp_state.append(ini_energy[k]-delta_e)
                else:
                    tmp_state.append(ini_energy[k])
            #需要进行归一化
            nor_energy = nor_state(tmp_state.copy())
            f2 = np.std(tmp_state)
            f2_nor = np.std(nor_energy)
            #print(nor_energy)
            #print(tmp_state)
            #print(f2)
            #print(f2_nor)
            cur_f = omega_*f1+(1-omega_)*f2_nor
            if f1 == 1.0:
                logger.info("f1:{0},f2:{1},f2_nor:{2},cur_f:{3},best_f:{4}".format(f1,f2,f2_nor,cur_f,best_f))
            if cur_f < best_f:
                best_f = cur_f
                best_f1 = f1
                best_f2_nor = f2_nor
                best_f2 = f2
                best_result = global_action.copy()
                logger.info("f1:{0},f2:{1},f2_nor:{2}cur_f:{3},best_f:{4},res:{5}".format(f1,f2,f2_nor,cur_f,best_f,best_result))

        
        #结束之后恢复初始能量，正式更新能量状态
        for k in range(n):
            robots[k].energy = ini_energy[k]

        update_energy2(best_result)
        e_state = []
        for k in range(n):
            e_state.append(robots[k].energy)

        end_time = time.perf_counter()
        res_becb_energy.append(ini_energy)
        res_afcb_energy.append(e_state)
        res_f2.append(best_f2)
        res_f2_nor.append(best_f2_nor)
        res.append(best_result)
        res_f1.append(best_f1)
        res_f.append(best_f)
        num_all.append(sum(best_result))
        time_all.append(end_time-start_time)
        print("episode "+str(epoch_i)+" finish!")
    

    res_np = np.array(res)
    res_f1_np = np.array(res_sll)
    res_f_np = np.array(res_f)
    res_becb_energy_np = np.array(res_becb_energy)
    res_afcb_energy_np = np.array(res_afcb_energy)
    res_f2_np = np.array(res_f2_nor)
    res_f2 = np.array(res_f2)
    num_all_np = np.array(num_all)
    time_np = np.array(time_all)
    np.save("./test/result",res_np)
    np.save("./test/f1",res_f1)
    np.save("./test/energy_becb",res_becb_energy_np)
    np.save('./test/energy_afcb',res_afcb_energy_np)
    np.save("./test/f2",res_f2_np)
    np.save("./test/f2_nor",res_f2_nor)
    np.save('./test/f',res_f_np)
    np.save('./test/num',num_all_np)
    np.save('./test/time',time_np)
    print(min(res_f1))
    print(sum(res_f1)/100)
    print(sum(res_f2_nor)/100)
    print(sum(res_f2)/100)
    print(sum(res_f)/100)
    print(num_all)
    print('model_test')
