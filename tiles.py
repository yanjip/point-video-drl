'''
@Author  ：Yan JP
@Created on Date：2023/5/6 19:58 
'''
import numpy as np
from para import *

np.random.seed(0)

def Pr_xk(i, r):
    s = 0
    for k in range(1, N_fovs + 1):
        s += k ** (-r)
    ans = i ** (-r)
    return ans / s

# M_all=np.array([300e6,600e6])
N_all=np. array([100,200])*1e3       #points
M_all_bound=np.array([10,20])*MB
# M_all = np.random.normal(15*MB, 10000000, size=10)

# for i in range(N):
N_points_size=np.random.randint(N_all[0],N_all[1],N)
# np.save('tiles_datasize.npy',M_data_size)
# print(M_data_size)

Fovs=[]
pr=[]

for i in range(N_fovs):
    Fovs.append(np.random.choice(N_points_size,N_F))
    a=Pr_xk(i+1,zipf)
    pr.append(a)
print(pr)
pr=np.array(pr)





class tile():
    def __init__(self):
        N_all = np.array([100, 200]) * 1e3  # points
        self.N_points_size = np.random.randint(N_all[0], N_all[1], N)
        self.M=M
        self.id=[i for i in range(N)]

        pass
    def get_Fov(self,):
        Fovs = []
        pr = []
        Fovs_tile_id=[]
        for i in range(N_fovs):
            fov_tile_id=np.random.choice(self.id,N_F)
            fov_tile_N_pts=[]
            for j in fov_tile_id:
                fov_tile_N_pts.append(self.N_points_size[j])
            Fovs.append(fov_tile_N_pts)
            Fovs_tile_id.append(fov_tile_id)
            a = Pr_xk(i + 1, zipf)
            pr.append(a)
        self.Fovs=np.array(Fovs)
        self.Fovs_tile_id=np.array(Fovs_tile_id)
        self.pr = np.array(pr)

    def get_dis(self):
        dis_range=[1,1.5,2.0]
        self.dis=np.random.choice(dis_range,N,p=[0.5,0.3,0.2])


    def get_O(self):  #生成NF个遮挡等级数
        # p=np.random.normal(0,2,NF)
        p = np.random.normal(0, 3, N)
        p = np.abs(p)
        p = np.clip(p, 0, 5)
        self.O = np.floor(p).tolist()
        return self.O

    def get_Qi(self,Ni,sumN,Oi,dis_i):
        k=-(sumN/Ni-1)/5
        b=sumN/Ni
        w_oi=k*Oi+b
        zi=w_oi*Ni/sumN

        return zi*(1/dis_i),zi

    # def get_zi(self,Ni,sumN,Oi):
    #     k=-(sumN/Ni-1)/5
    #     b=sumN/Ni
    #     w_oi=k*Oi+b
    #     zi=w_oi*Ni/sumN
    #
    #     return zi

    def get_Q(self,):
        Q=[]
        z=[]
        sumN=sum(self.N_points_size)
        for i in range(N):
            q,zi=self.get_Qi(N_points_size[i],sumN,self.O[i],self.dis[i])
            Q.append(q)
            z.append(zi)
        self.Q=np.array(Q)
        self.z=np.array(z)

ttile=tile()
ttile.get_Fov()
ttile.get_O()
ttile.get_dis()
ttile.get_Q()
print(ttile)

# import pickle
# out_put = open("tiles.pkl", 'wb')
# t = pickle.dumps(ttile)
# out_put.write(t)
# out_put.close()
