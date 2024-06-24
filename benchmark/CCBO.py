"""
对比方法，基于环的选择
2019-《JSSA:joint sidelobe suppression approach for collaborative beamforming in wireless:sensor networks》
"""
from turtle import pos
import sympy as sy
import numpy as np
import math
import matplotlib.pyplot as plt
import time

#global parameters
test_num = 1
lambda_ = 5
phi = sy.Symbol('phi')
phi0 = 0.0
n = 10
global robots 
robots = []
val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.05)
global idea_position
idea_position = []
global best_solution
best_solution = []
global run_time_list 
run_time_list = []
global sll_list 
sll_list = []



class Robot(object):
    def __init__(self,id:int=None,
                       r:int=None,
                       local_phi:float=None):
        self.id = id
        self.r = r
        self.phi = local_phi
        self.alpha = 2*sy.pi/lambda_*self.r*(sy.cos(phi0-self.phi)-sy.cos(phi-self.phi))

#初始化机器人的位置
def reset(cur_position):
    robots.clear()
    #print("current position")
    for i in range(n):
        tmp = Robot(i,cur_position[i][0],cur_position[i][1])
        robots.append(tmp)


#计算理想的节点位置
def compute_position():
    idea_position.clear()
    a = 25
    b = 25*math.sqrt(3)
    c = 12.5
    d = 12.5*math.sqrt(3)
    #circle1
    idea_position.append([a,0])
    idea_position.append([c,d])
    idea_position.append([-c,d])
    idea_position.append([-25,0])
    idea_position.append([-c,-d])
    idea_position.append([c,-d])
    
    #circle2
    idea_position.append([50,0])
    idea_position.append([b,a])
    idea_position.append([a,b])
    idea_position.append([0,50])
    idea_position.append([-a,b])
    idea_position.append([-b,a])
    idea_position.append([-50,0])
    idea_position.append([-b,-a])
    idea_position.append([-a,-b])
    idea_position.append([0,-50])
    idea_position.append([a,-b])
    idea_position.append([b,-a])

def get_sll(x):
    num = 0
    max_val = 0.0
    f_x = 0
    f_y = 0
    num = len(x)

    for i in range(num):
        f_x += sy.cos(robots[x[i]].alpha)
        f_y += sy.sin(robots[x[i]].alpha)
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

        if left_phi <= right_phi:
            val_phi = np.arange(left_phi,right_phi+0.01,0.05)
            for i in val_phi:
                max_val = max(max_val,(dis.subs(phi,i)).evalf(n=5))
        return max_val
    else:
        return 1.0


def compute_distance():
    pos_x = []
    pos_y = []
    for i in range(n):
        pos_x.append(robots[i].r*math.cos(robots[i].phi))
        pos_y.append(robots[i].r*math.sin(robots[i].phi))

    dis = {}
    for i in range(n):
        min_dis = 1000
        for j in range(18):
            cur_dis = math.sqrt((pos_x[i]-idea_position[j][0])**2+(pos_y[i]-idea_position[j][1])**2)
            min_dis = min(cur_dis,min_dis)
        dis[i] = min_dis
    dis = sorted(dis.items(),key = lambda x:x[1])
    return dis
    

if __name__=='__main__':
    
    print('begin')
    position = np.load('../pos_10/pos_polar.npy').tolist()
    num_list1 = np.load('num.npy').tolist()
    print(num_list1)
    compute_position()
    for k in range(100):
        print("test {0} begin!".format(k))
        reset(position[k])
        begin_time = time.perf_counter()
        dis = compute_distance()
        solution = []
        num_tmp = num_list1[k]
        for i in range(num_tmp):
            solution.append(dis[i][0])
        sll = get_sll(solution)
        #print(solution)
        res= []
        for i in range(n):
            res.append(0)
            if i in res:
                res[i] = 1
        best_solution.append(res)
        end_time = time.perf_counter()
        run_time = end_time-begin_time
        run_time_list.append(run_time)
        sll_list.append(sll)
        print("test {0} finish!".format(k))
    
    best_solution_np = np.array(best_solution)
    sll_list_np = np.array(sll_list)
    run_time_list_np = np.array(run_time_list)
    np.save("ring_res1",best_solution)
    np.save("ring_sll1",sll_list_np)
    np.save("ring_time1",run_time_list_np)
    