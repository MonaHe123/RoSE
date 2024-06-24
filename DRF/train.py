"""
to modify
(1)parameters
(2)state encoding
(3)reward function

f2
以节点的能量为标准
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
    def reset(self):
        #考虑的范围是100m*100m的范围
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
        max_old_action = max(global_action)
        max_new_action = max(max_old_action,action)
        tmp_action[self.id] = action
        #如果没有机器人参与
        if max_new_action == 0:
            return -1

        #AF部分的奖励
        f1 = get_sll(self.id,action)
        r1 = -f1

        #energy部分
        #采取动作之后会导致能量的改变，但是此时不是状态更新，所以不能改变全局的量
        e_state = []
        cur_num = sum(tmp_action)
        delta_e = 0.0
        if action == 1:
            delta_e = l_data*(e_cct+e_tx*((dis_controller)**path_loss)/(cur_num**2))
        cur_energy = self.energy-delta_e
        for i in range(n):
            if i!=self.id:
                e_state.append(robots[i].energy-cur_energy)
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
        r = omega_*r1+(1-omega_)*r2
        logger.info("f1:{0},f2:{1},r:{2}".format(f1,f2,r))
        return round(r,5)
    

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
def get_sll(id,action):
    num = sum(global_action)
    fx = []
    fy = []
    if action != global_action[id]:
        if action == 1:
            num += 1
            for j in range(len(f_x)):
                fx.append(0.0)
                fy.append(0.0)
                fx[j]=f_x[j]+sy.cos(robots[id].alpha_value[j])
                fy[j]=f_y[j]+sy.sin(robots[id].alpha_value[j])
        else:
            num -= 1
            for j in range(len(f_x)):
                fx.append(0.0)
                fy.append(0.0)
                fx[j]=f_x[j]-sy.cos(robots[id].alpha_value[j])
                fy[j]=f_y[j]-sy.sin(robots[id].alpha_value[j])
    else:
        fx = f_x.copy()
        fy = f_y.copy()
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
    print('left:{0},right:{1}'.format(left,right))
    if max_val == 0:
        return 1
    else:
        return max_val

if __name__=='__main__':
    logger = logging.getLogger(__name__)
    current_date_time = datetime.now()
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    fh = logging.FileHandler(f"./train_res/log.txt")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.basicConfig(level=logging.NOTSET)
    logger.info("DRF W=0.2 NODE")

    max_episode = 1200
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
    episode_model_param= 1100

    score = []
    #step = []
    #每个机器人的av_reward
    av_reward = []
    s0 = []
    a0 = []
    #每个机器人的每步的reward
    reward = []

    for i in range(n):
        robots.append(Robot(id=i))
        robots[i].reset_file()
        score.append(0.0)
        av_reward.append(0.0)
        global_action.append(1)
        tmp_action.append(1)
        s0.append(0)
        a0.append(1)
        reward.append(0.0)
    ini_diff()
    for i in range(len(val)):
        f_x.append(0)
        f_y.append(0)
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
            score[i] = 0.0
            global_action[i] = 1
            tmp_action[i] = 1
        #ini_diff()
        compute_diff()
        for i in range(n):
            robots[i].state = robots[i].new_state()
        
        #robo_info()
        
        for j in range(max_step):
            #print("step "+str(j)+" start!")
            logger.info("step "+str(j)+" start!")
            step_count += 1
            get_f()
            for k in range(n):
                s0[k] = robots[k].state
                a0[k] = robots[k].Q_net.sample_action(s0[k],epsilon)
                reward[k] = robots[k].act(a0[k])
                score[k] += reward[k]
                #print("robot {0} take action {1}, and reward is {2}".format(k,a0[k],reward[k]))
                #print("global action:{0}".format(global_action))
                #print("tmp_action:{0}".format(tmp_action))
                logger.info("robot {0} take action {1}, and reward is {2}".format(k,a0[k],reward[k]))
                #logger.info("global action:{0}".format(global_action))
                #logger.info("tmp_action:{0}".format(tmp_action))
                if step_count > train_begin:
                    train_flag = True
                    train_net(robots[k].Q_net,robots[k].Q_target,robots[k].optimizer,robots[k].losses,1,gamma,batch_size)
                    #print("start trainning")

            #每个机器人都采取了一个步骤
            #交换参数的时候，考虑的是所有的机器人，所以所有训练结束之后合并
            #当机器人的数量很大的时候，或许可以只考虑一部分
            #根据全局的动作更新状态
            #global_action = tmp_action
            for k in range(n):
                global_action[k] = tmp_action[k]
            cur_num = sum(global_action)
            delta_e = 0.0
            if cur_num != 0:
                delta_e = l_data*(e_cct+e_tx*((dis_controller)**path_loss)/(cur_num**2))
            for k in range(n):
                if a0[k] == 1:
                    robots[k].energy -= delta_e
                robots[k].state = robots[k].new_state()
                s1 = robots[k].state
                robots[k].Q_net.save_trans((s0[k],a0[k],reward[k],s1))
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
            av_reward[i] += score[i]/max_step
            robots[i].av_reward.append(av_reward[i]/(epoch_i+1))
            robots[i].ep_reward.append(score[i]/max_step)
            #一个epoch_info保存10个条目，每个对应一个机器人，具体内容目前位置的av_reward和ep_av_reward
            ep_info.append([av_reward[i]/(epoch_i+1),score[i]/max_step])
        ep_info_np = np.array(ep_info)
        np.save('./train_res/ep_info/episode_'+str(epoch_i),ep_info_np)

        print("episode "+str(epoch_i)+" finish!")

    #储存每个robot的平均reward
    for i in range(n):
        av_reward_np = np.array(robots[i].av_reward)
        print(robots[i].av_reward)
        np.save(robots[i].AV_info+"av_reward.npy",av_reward_np)
        ep_reward_np = np.array(robots[i].ep_reward)
        print(robots[i].ep_reward)
        np.save(robots[i].AV_info+"ep_reward.npy",ep_reward_np)
        print("save reward")
    
    #首先画出每个机器人的reward图
    for i in range(n):
        plot_curve(robots[i].av_reward,i)
    p = fig.add_subplot(row,col,11)
    x = []
    for i in range(len(robots[0].av_reward)):
        x.append(i)

    for i in range(n):
        p.plot(x,robots[i].av_reward,'-')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
    plt.legend(["All Robot of reward 1"])
    plt.savefig("reward1.pdf")
    plt.grid()
    plt.show()
        #奖励曲线和损失曲线