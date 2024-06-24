"""
to modify
(1)parameters
(2)state encoding
(3)reward function

f2:以节点为中心
reward function:相同的奖励
所有的机器人采取动作之后获得奖励
"""
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
        self.model_path = "./train_res/model_"+str(self.id)+"/DDQN"
        self.AV_info = "./train_res/info_"+str(self.id)+"/"
        #包括位置状态和能量状态,2*(n-1)
        self.input_size = 2*(n-1)
        self.outputsize = 2
        self.mem_len = 30000
        self.Q_net = DQN(self.input_size,self.outputsize,self.mem_len)
        self.Q_target = DQN(self.input_size,self.outputsize,self.mem_len)
        self.Q_target.load_state_dict (self.Q_net.state_dict())
        self.optimizer = optim.Adam(self.Q_net.parameters(),lr=learning_rate)
        self.losses = nn.MSELoss()
        self.reset_file()

    def reset_file(self):
        ensure_dir(self.model_path)
        #ensure_dir(self.AV_info)


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
    def reset(self):
        #考虑的范围是400m*400m的范围
        self.r = np.random.randint(1,51)
        self.phi = round(np.random.uniform(0,2*math.pi),4)
        self.energy = ini_energy
        self.alpha = self.get_alpha()
        self.alpha_value = []
        #为了学习更加有代表性，因此进行一些归一化的操作，即将角度都规约到0到2*pi
        for i in val:
            self.alpha_value.append(((self.alpha.subs(phi,i))%(2*sy.pi)).evalf(n=5))


    #采取动作
    def act(self,action):
        global_action[self.id] = action
    
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
              
def robo_info():
    logger.info("robot information")
    logger.info("#"*10)
    for i in range(n):
        logger.info("robot:{0}".format(i))
        logger.info("r:{0},phi:{1},energy:{2},state:{3}".format(robots[i].r,robots[i].phi,robots[i].energy,robots[i].state))
    logger.info("#"*10)



if __name__=='__main__':
    logger = logging.getLogger(__name__)
    current_date_time = datetime.now()
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    fh = logging.FileHandler(f"./train_res/log_250_100.txt")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.basicConfig(level=logging.NOTSET)
    logger.info("ERF W=0.5 NODE")

    max_episode = 300
    max_step = 100
    gamma = 0.95
    step_count = 0
    train_flag = False
    mem_len = 30000
    train_begin = 150
    batch_size = 128
    epsilon = 0.12535073493989754
    log_num = 5
    C = 100
    log_start = 5
    episode_model_param = 250


    score = 0.0
    #step = []
    #每个机器人的av_reward
    av_reward = []
    sum_av_reward = 0.0
    s0 = []
    a0 = []
    #每个机器人的每步的reward
    reward = []
    ep_reward = []


    for i in range(n):
        robots.append(Robot(id=i))
        robots[i].reset_file()
        global_action.append(1)
        tmp_action.append(1)
        s0.append(0)
        a0.append(1)
        reward.append(0.0)
    ini_diff()
    #tmp_action = global_action
    for i in range(n):
        robots[i].Q_net = torch.load(robots[i].model_path+str(episode_model_param)+".pth")
        robots[i].Q_target.load_state_dict(robots[i].Q_net.state_dict())
        print("load model")
    for epoch_i in range(episode_model_param,max_episode):
        #print("eposide "+str(epoch_i)+" start!")
        logger.info("eposide "+str(epoch_i)+" start!")
        for i in range(n):
            robots[i].reset()
            global_action[i] = 1
            tmp_action[i] = 1
        #ini_diff()
        score = 0.0
        compute_diff()
        for i in range(n):
            robots[i].state = robots[i].new_state()
        
        #robo_info()
        
        for j in range(max_step):
            #print("step "+str(j)+" start!")
            logger.info("step "+str(j)+" start!")
            step_count += 1
            for k in range(n):
                s0[k] = robots[k].state
                a0[k] = robots[k].Q_net.sample_action(s0[k],epsilon)
                robots[k].act(a0[k])
                #print("robot {0} take action {1}, and reward is {2}".format(k,a0[k],reward[k]))
                #print("global action:{0}".format(global_action))
                #print("tmp_action:{0}".format(tmp_action))
                
                #logger.info("global action:{0}".format(global_action))
                #logger.info("tmp_action:{0}".format(tmp_action))
                if step_count > train_begin:
                    train_flag = True
                    train_net(robots[k].Q_net,robots[k].Q_target,robots[k].optimizer,robots[k].losses,1,gamma,batch_size)
                    #print("start trainning")


            max_action = max(global_action)
            cur_num = sum(global_action)
            delta_e = 0.0
            if cur_num != 0:
                delta_e = l_data*(e_cct+e_tx*((dis_controller)**path_loss)/(cur_num**2))
            e_state = []
            r = 0.0
            for k in range(n):
                if a0[k] == 1:
                    robots[k].energy -= delta_e
                e_state.append(robots[k].energy)
            if max_action == 0:
                r = -1
            else:
                f1 = get_sll()
                r1 = -f1
                min_e = min(e_state)
                max_e = max(e_state)
                #归一化
                gap_e = max_e-min_e
                if gap_e != 0:
                    for i in range(len(e_state)):
                        e_state[i] = (e_state[i]-min_e)/gap_e
                    f2 = np.std(e_state)
                else:
                    f2 = 0.0
                r2 = -f2
                r = round(omega_*r1+(1-omega_)*r2,5)
                logger.info("f1:{0},f2:{1},reward:{2}".format(f1,f2,r))
            score += r
            for k in range(n):
                robots[k].state = robots[k].new_state()
                s1 = robots[k].state
                reward[k] = r
                robots[k].Q_net.save_trans((s0[k],a0[k],reward[k],s1))
                logger.info("robot {0} take action {1}".format(k,a0[k]))
                #print("new state of robot",end=" ")
                #print(k,end=" :")
                #print(s1)
            #robo_info()
            
            if step_count % C == 0 and train_flag == True:
                print("change!")
                sum_p = robots[0].Q_net.state_dict()
                for k in range(1,n):
                    for key in sum_p.keys():
                        sum_p[key] += robots[k].Q_net.state_dict()[key]
                for key in sum_p.keys():
                    sum_p[key] /= n
                
                for k in range(n):
                    robots[k].Q_target.load_state_dict(sum_p)
            #print("step "+str(j)+" finish!")
        
        #每个epoch结束，记录学习的信息
        #记录每个episode的奖励信息
        ep_info = []
        for i in range(n):
            if epoch_i > log_start and ((epoch_i+1)%log_num == 0 or ((epoch_i+1)== max_episode)) and train_flag == True:
                torch.save(robots[i].Q_net,robots[i].model_path+str(epoch_i+1)+".pth")
                print("save model")
            sum_av_reward += score/max_step
            av_reward.append(sum_av_reward/(epoch_i+1))
            ep_reward.append(score/max_step)
            #一个epoch_info保存10个条目，每个对应一个机器人，具体内容目前位置的av_reward和ep_av_reward
            ep_info.append([sum_av_reward/(epoch_i+1),score/max_step])
        ep_info_np = np.array(ep_info)
        np.save('./train_res/ep_info/episode_'+str(epoch_i),ep_info_np)

        print("episode "+str(epoch_i)+" finish!")


    av_reward_np = np.array(av_reward)
    np.save('av_reward',av_reward_np)
    