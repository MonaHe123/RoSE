import math
import numpy as np
import scipy.special as sc_special
import sympy as sy
import matplotlib.pyplot as plt
import time

#global paprameters
alpha_ = 0.01
beta_ = 1.5
yita_ = 0.9
max_iter = 50
p_a = 0.25
global nests
nests = []
#N_pop
global n
n = 10
#机器人的数量
global m
m = 10


#有关CB的参数
lambda_ = 0.125
global phi 
phi = sy.Symbol('phi')
global phi0
phi0 = 0.0
global robots
robots = []
global val 
val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.1)    
global robot_r
robot_r = []
global robot_a
robot_a = []  

#有关能量的参数
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
global k1
k1 = 7.4
global k2
k2 = 0.29
global omega_
omega_ = 0.5


#进行levy飞行，返回L_step
#规模是N*M,也就是n个解，每个解包括m个机器人的选择
def levy_flight():
    sigma_u = (sc_special.gamma(1+beta_)*np.sin(np.pi*beta_/2)/(sc_special.gamma((1+beta_)/2)*beta_*(2**((beta_-1)/2))))**(1/beta_)
    sigma_v = 1
    u = np.random.normal(0,sigma_u,(n,m))
    v = np.random.normal(0,sigma_v,(n,m))
    steps = (u/((np.abs(v))**(1/beta_)))*alpha_
    steps = steps.tolist()
    return steps

def generate_nests():
    nests.clear()
    for i in range(n):
        nests.append([])
        #print("HELLO")
        for j in range(m):
            nests[i].append(np.random.randint(0,2))
            #print(nests[i][j])
            #print("oh")
    return nests


def abandon_nests(x):
    rand = np.random.rand()
    if rand<=p_a:
        p = np.random.randint(0,n)
        q = np.random.randint(0,n)
        nest_tmp = []
        fit = []
        nest_tmp.append(nests[p].copy())
        fit.append(get_fit(nest_tmp[0]))
        nest_tmp.append(nests[q].copy())
        fit.append(get_fit(nest_tmp[1]))
        x_cross = []
        len1 = int(m/2)
        for i in range(len1):
            x_cross.append(nest_tmp[0][i])
        for i in range(m-len1):
            x_cross.append(nest_tmp[1][i+len1])
        nest_tmp.append(x_cross)
        fit.append(get_fit(nest_tmp[2]))
        min_index = fit.index(min(fit))
        return nest_tmp[min_index]
    else:
        return x
    
#定义robot类
class Robot(object):
    def __init__(self,id:int=None):
        self.id = id
        self.r = None
        self.phi = None
        self.alpha = None
        #初始的时候，机器人的能量为初始能量
        self.energy = ini_energy

    def get_alpha(self):
        return 2*sy.pi/lambda_*self.r*(sy.cos(phi0-self.phi)-sy.cos(phi-self.phi))
    
    def reset(self):
        #self.r = np.random.randint(1,11)
        #self.phi = round(np.random.uniform(0,2*math.pi),4)
        self.r = robot_r[self.id]
        self.phi = robot_a[self.id]
        self.alpha = self.get_alpha()
        self.alpha_value = []
        #为了学习更加有代表性，因此进行一些归一化的操作，即将角度都规约到0到2*pi
        for i in val:
            self.alpha_value.append(((self.alpha.subs(phi,i))%(2*sy.pi)).evalf(n=5))
    

global f_x
f_x = []
global f_y 
f_y = []

#计算所有的alpha的和，其实就是AF的值
def get_sll(x):
    for i in range(len(val)):
        f_x[i] = 0.0
        f_y[i] = 0.0
        #sum_alpha[i] = 0
        for j in range(m):
            if x[j] == 1:
                    f_x[i] += sy.cos(robots[j].alpha_value[i])
                    f_y[i] += sy.sin(robots[j].alpha_value[i])
    num = sum(x)
    f = []
    for i in range(len(val)):
        f.append(sy.sqrt(f_x[i]**2+f_y[i]**2))
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
    #print(max_val)
    if max_val == 0:
        return 1
    else:
        return max_val

def get_energy(x):
    num = sum(x)
    max_energy = 0
    min_energy = ini_energy
    if num == 0:
        return 1.0
    delta_e = l_data*(e_cct+e_tx*((dis_controller)**path_loss)/(num**2))
    energy = []
    #m为机器人的数量
    #得到机器人的能量分布
    for i in range(m):
        energy.append(robots[i].energy-x[i]*delta_e)
        print('robot:{0},energy:{1}'.format(i,robots[i].energy))
        if energy[i] > max_energy:
                max_energy = energy[i]
        if energy[i] < min_energy:
                min_energy = energy[i]
    for i in range(m):
        energy[i] = (energy[i]-min_energy)/(max_energy-min_energy)
    f2_nor = np.std(energy)
    return f2_nor

def get_fit(x):
    f1 = get_sll(x)
    f2 = get_energy(x)
    f = omega_*f1+(1-omega_)*f2
    return f

def cukcoo_search():
    x_best = None
    min_fit = 1.0
    generate_nests()
    for iter in range(max_iter):
        print("iter:{0}".format(iter))
        steps = levy_flight()
        for i in range(n):
            #print("the {0} solution".format(i))
            rand = np.random.rand()
            l_step = steps[i]
            if rand <= yita_:
                for j in range(m):
                    sig_1=1/(1+math.exp(-l_step[j]))
                    #rand1 = np.random.rand()
                    if rand <= sig_1:
                        nests[i][j] = 1
                    else:
                        nests[i][j] = 0
            else:
                for j in range(m):
                    #rand1 = np.random.rand()
                    if l_step[j] <= 0:
                        sig_2 = 1-(2/(1+math.exp(-l_step[j])))
                        if rand <= sig_2:
                            nests[i][j] = 0
                    else:
                        sig_2 = 2/(1+math.exp(-l_step[j]))-1
                        if rand <= sig_2:
                            nests[i][j] = 1
            rand_index = np.random.randint(0,n)
            x_i = nests[i].copy()
            x_j = nests[rand_index].copy()
            fit_i = get_fit(x_i)
            fit_j = get_fit(x_j)
            if fit_j < fit_i:
                nests[i] = nests[rand_index].copy()
            x_ = abandon_nests(x_i)
            nests[i] = x_.copy()
            #find the best
            fit = get_fit(nests[i])
            if fit < min_fit:
                min_fit = fit
                x_best = nests[i].copy()
                #print('best sll:{0}'.format(min_sll))
                #print('best result:{0}'.format(x_best))
        print("iter:{0} finish".format(iter))
        #print('best sll:{0}'.format(min_sll))
        #print('best result:{0}'.format(x_best))
    return x_best,min_fit

if __name__=='__main__':
    time_slot = 100
    position = np.load('../pos_polar.npy').tolist()
    motion_time = np.load('../time.npy').tolist()
    velocity = np.load('../velocity.npy').tolist()
    #print(len(motion_time))
    #print(len(velocity))
    #print(len(velocity[0]))
    
    cukcoo_fit = []
    cuckoo_time = []
    x_best = []
    robots = []
    cuckoo_f1 = []
    cuckoo_f2 = []
    e_state = []
    for i in range(m):
        robots.append(Robot(i))
    for i in range(len(val)):
        f_x.append(0)
        f_y.append(0)
    for k in range(0,time_slot):
        print("time slot {0} begin!".format(k))
        #初始化所有机器人的位置
        robot_r.clear()
        robot_a.clear()
        for j in range(m):
            robot_r.append(position[k][j][0])
            robot_a.append(position[k][j][1])
            #print(len(robot_r))
            #print(len(robot_a))
        for j in range(m):
            robots[j].reset()
            robots[j].energy -= motion_time[k]*(k1*velocity[k][j]+k2)
            #print("robot {0}:({1},{2},{3})".format(robots[j].id,robots[j].r,robots[j].phi,robots[j].energy))
        
        start_time = time.perf_counter()
        tmp,tmp_fit = cukcoo_search()
        num = sum(tmp)
        delta_e = l_data*(e_cct+e_tx*((dis_controller)**path_loss)/(num**2))
        e_state = []
        for j in range(m):
            if tmp[j] == 1:
                robots[j].energy -= delta_e
            e_state.append(robots[j].energy)
        #print(e_state)
        f1 = get_sll(tmp)
        f2 = np.std(e_state)
        end_time = time.perf_counter()
        all_time = end_time-start_time
        #print("res:{0}".format(tmp))
        #print('sll:{0}'.format(tmp_sll))
        cukcoo_fit.append(tmp_fit)
        x_best.append(tmp)
        cuckoo_f1.append(f1)
        cuckoo_f2.append(f2)
        cuckoo_time.append(all_time)
        tmp_res = []
        tmp_res.append(f1)
        tmp_res.append(f2)
        tmp_res.append(tmp_fit)
        tmp_res.append(all_time)
        tmp_res.append(e_state)
        print(tmp_res)
        tmp_res_np = np.array(tmp_res)
        np.save("./ep_info/res"+str(k),tmp_res_np)
        print("time slot {0} finish!".format(k))
    
    #储存结果
    cukcoo_fit_np = np.array(cukcoo_fit)
    x_best_np = np.array(x_best)
    cuckoo_time_np = np.array(cuckoo_time)
    cuckoo_f1_np = np.array(cuckoo_f1)
    cuckoo_f2_np = np.array(cuckoo_f2)
    np.save('./ep_info/cukcoo_fit',cukcoo_fit_np)
    np.save("./ep_info/cukcoo_selection",x_best_np)
    np.save("./ep_info/cukcoo_time",cuckoo_time_np)
    np.save("./ep_info/cukcoo_f1",cuckoo_f1_np)
    np.save("./ep_info/cukcoo_f2",cuckoo_f2_np)
    ave = sum(cukcoo_fit)/time_slot
    print("the average fit is {0}".format(ave))
