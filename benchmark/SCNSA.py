"""
对比方法，基于直线的选择
2016-《sidelobe control by node selection algorithm based on virtual linear array for collaborative beamforming in WSNs》
"""

import sympy as sy
import numpy as np
import math
import matplotlib as plt
import time

#global paramters
test_num = 100      #总的测试的次数
lambda_ = 5
phi = sy.Symbol('phi')
phi0 = 0.0
n = 10
global robots
robots = []
val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.05)
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

   


def get_sll(x):
    num = 0
    max_val = 0.0
    f_x = 0
    f_y = 0
    num = len(x)
    #如果当前没有节点参与CB，那么得到的是0.0，否则就是大于等于0的
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
        #那么max SLL的范围就是从left_phi到right_phi
        #val = np.arange(round(math.pi/30,4),round(71/36*math.pi,4),0.05)  
        if left_phi <= right_phi:
            val_phi = np.arange(left_phi,right_phi+0.01,0.05)
            for i in val_phi:
                max_val = max(max_val,(dis.subs(phi,i)).evalf(n=5))
        return max_val
    else:
        return 1.0

#返回当前的机器人距离直线的距离
def get_dis(A,B,C):
    pos_x = []
    pos_y = []
    for i in range(n):
        pos_x.append(robots[i].r*math.cos(robots[i].phi))
        pos_y.append(robots[i].r*math.sin(robots[i].phi))
    dis = {}
    for i in range(n):
        dis[i] = (abs(A*pos_x[i]+B*pos_y[i]+C))/(math.sqrt(A**2+B**2))
    dis = sorted(dis.items(),key = lambda x:x[1])
    return dis


if __name__ == '__main__':
    position = np.load('../pos_10/pos_polar.npy').tolist()
    num = np.load("num.npy").tolist()
    print(num)
    #随机产生源节点，然后根据该节点选择接近直线的num个节点参与CB
    #目标位置
    d_t = 1000
    for k in range(100):
        print("test {0} begin!".format(k))
        #随机产生源节点
        origin = np.random.randint(0,10)
        #初始化机器人的位置
        reset(position[k])
        begin_time = time.perf_counter()
        #确定源和目标位置的坐标
        x_s = position[k][origin][0]*math.cos(position[origin][0][1])
        y_s = position[k][origin][0]*math.sin(position[origin][0][1])
        x_base = x_s-math.sqrt((d_t)**2-(y_s)**2)
        if y_s != 0:
            k_ = (y_s)/(x_s-x_base)
            A = -1/k_
            B = -1
            C = -A*x_s+y_s
        else:
            A = 1
            B = 0
            C = -(x_s+x_base)/2
        dis = get_dis(A,B,C)
        #print(dis)
        participate = []
        for i in range(num[k]):
            participate.append(dis[i][0])
        #print(participate)
        sll = get_sll(participate)
        #print(sll)
        sll_list.append(sll)
        end_time = time.perf_counter()
        run_time_list.append(end_time-begin_time)
        
        res = []
        for i in range(n):
            res.append(0)
            if i in participate:
                res[i] = 1
        best_solution.append(res)
        print("test {0} finish!".format(k))
    #print(best_solution)
    print(best_solution)
    best_np = np.array(best_solution)
    time_list_np = np.array(run_time_list)
    sll_list_np = np.array(sll_list)
    np.save("line_res",best_np)
    np.save("line_time",time_list_np)
    np.save("line_sll",sll_list_np)

    


