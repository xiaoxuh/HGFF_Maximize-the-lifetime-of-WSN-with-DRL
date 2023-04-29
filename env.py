import math
import random
import time
import numpy as np


class node:
    def __init__(self,x,y,e0,e1):
        self.x=x
        self.y=y
        self.e0=e0
        self.e1=e1
        self.e0_e1=1
        self.c=0
        self.send=0
        self.rece=0
        self.sensor_rate=1
class sink:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.feasible_flag=1
class site:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.feasible_flag=1

class wsn_env:
    def __init__(self,dyn,map_type,map_name):
        # random initialize or not
        self.dyn=dyn
        self.map_type = map_type
        self.map=map_name
        self.num_nodes=0
        self.num_sites=0
        self.site_width=0
        self.site_length=0
        self.site_map=[]
        self.node_map=[]
        self.life_time=0

        self.nodes=[]
        self.sites=[]
        self.e_initial = 2e10
        self.max_x=0.0
        self.max_y=0.0
        # sink
        self.s=sink(0,0)
        self.pos_normalization=10
        self.c_normalization=1e5
        # transmission range
        self.R=30
        self.A = 50
        self.B = 100
        self.V = 2
        # current FA cost between all nodes in all maps
        self.cost=[[[]]]
        self.e_receive=50.0
        self.SENSOR_DATA=1.0
        self.state_size_w, self.state_size_l =0,0
        self.DT=3600
    #     mark
        self.mark_sink=0
        self.mark_site=1
        self.mark_nodes=2
    # adj
        self.matrix = []
    #     gaussian
        self.phase=0
        self.u,self.v=0.0,0.0
        self.pi=3.141592654
    #     time
        self.start=0.0
        self.orginal_x=[]
        self.orginal_y=[]

        np.set_printoptions(threshold=np.inf)
    def read_data(self):

        prefix='data/'+ 'type_' + str(self.map_type)+ '_'
        path = prefix + str(self.map) + '.txt'
        self.start=time.time()
        self.site_map=[]
        self.node_map=[]
        with open(path,'r') as f:
            lines = f.readlines()
            flag=True
            for line in lines:
                a=line.split()
                if flag:
                    self.site_width=int(a[0])
                    self.site_length=int(a[1])
                    self.num_nodes=int(a[2])
                    self.num_sites=self.site_width*self.site_length
                    flag=False
                    continue

                if not flag:
                    if len(a)==3:
                        self.site_map.append([int(a[0]),int(a[1]),int(a[2])])
                    else:
                        self.node_map.append([float(a[0]),float(a[1])])


    def build_graph(self):
        self.matrix = []
        for a in range(self.num_sites):
            self.matrix.append([0 for m in range(self.num_sites+self.num_nodes)])
            for b in range(self.num_sites):
                dis_sites_sites = math.sqrt((self.sites[a].x - self.sites[b].x) ** 2 + (self.sites[a].y - self.sites[b].y) ** 2)
                if dis_sites_sites<self.R:
                    self.matrix[a][b]=dis_sites_sites/self.pos_normalization
            for c in range(self.num_nodes):
                dis_site_nodes=math.sqrt((self.sites[a].x - self.nodes[c].x) ** 2 + (self.sites[a].y - self.nodes[c].y) ** 2)
                if dis_site_nodes<self.R:
                    self.matrix[a][c+self.num_sites]=dis_site_nodes/self.pos_normalization

        for i in range(self.num_nodes):
            self.matrix.append([0 for n in range(self.num_sites + self.num_nodes)])

            for j in range(self.num_sites):
                dis_node_sites = math.sqrt((self.nodes[i].x - self.sites[j].x) ** 2 + (self.nodes[i].y - self.sites[j].y) ** 2)
                if dis_node_sites<self.R:
                    self.matrix[i+self.num_sites][j]=dis_node_sites/self.pos_normalization
            for k in range(self.num_nodes):

                dis_nodes_nodes = math.sqrt((self.nodes[i].x - self.nodes[k].x) ** 2 + (self.nodes[i].y - self.nodes[k].y) ** 2)
                if dis_nodes_nodes<self.R:
                    self.matrix[i+self.num_sites][k+self.num_sites]=dis_nodes_nodes/self.pos_normalization

    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.life_time = 0
        self.read_data()
        min_d=float(1e50)
        init_g=-1
        self.nodes = []
        self.sites = []
        init_state = []
        for i in range(self.num_nodes):
            node_x,node_y=self.node_map[i][0], self.node_map[i][1]
            self.nodes.append(node(node_x,node_y,self.e_initial,self.e_initial))
            if node_x>self.max_x:
                self.max_x=node_x
            if node_y>self.max_y:
                self.max_y=node_y

        self.s.x,self.s.y=self.max_x/2,self.max_y/2

        for j in range(self.num_sites):
            site_x, site_y = self.site_map[j][0], self.site_map[j][1]
            self.sites.append(site(site_x, site_y))
            init_state.append([self.mark_site,site_x/self.pos_normalization, site_y/self.pos_normalization,0,0])
            dis = math.sqrt(math.pow(site_x - self.s.x, 2) + math.pow(site_y - self.s.y, 2))

            if self.sites[j].feasible_flag and dis<min_d:
                min_d=dis
                init_g=j

        self.s.x,self.s.y=self.sites[init_g].x, self.sites[init_g].y

        init_state[init_g][0]=self.mark_sink

        self.flow_routing()
        for i in range(self.num_nodes):
            self.nodes[i].e1-=self.nodes[i].c*self.DT
            self.nodes[i].e0_e1=self.nodes[i].e0/self.nodes[i].e1
            init_state.append(
                [self.mark_nodes,self.nodes[i].x / self.pos_normalization, self.nodes[i].y  / self.pos_normalization, self.nodes[i].e1 / self.nodes[i].e0,
                 self.nodes[i].c / self.c_normalization])
            self.orginal_x.append(self.nodes[i].x)
            self.orginal_y.append(self.nodes[i].y)
        self.build_graph()
        self.matrix=np.array(self.matrix)
        init_state=np.vstack((list(map(list,zip(*init_state))),self.matrix))
        self.state_size_w,self.state_size_l=len(init_state),len(init_state[0])

        return init_state


    def isdead(self):
        dead=False
        for i in range(self.num_nodes):
            if (self.nodes[i].e1<0):
                dead=True
        return dead

    def update_dyn_network(self, max_x, max_y):

        for i in range(self.num_nodes):
            self.nodes[i].x=np.clip(np.random.normal(self.orginal_x[i],3),0,max_x)
            self.nodes[i].y=np.clip(np.random.normal(self.orginal_y[i],3),0,max_y)

    def flow_routing(self):

        used=[0 for i in range(self.num_nodes)]
        parent=[0 for i in range(self.num_nodes+1)]
        shortest_dis=[1e50 for i in range(self.num_nodes)]
        self.cost = [[1e50 for a in range(self.num_nodes)] for b in range(self.num_nodes)]

        for i in range(self.num_nodes):
            e_ij, dis = 0.0, 0.0
            for j in range(self.num_nodes):
                dis=math.sqrt((self.nodes[i].x-self.nodes[j].x)**2+(self.nodes[i].y-self.nodes[j].y)**2)
                if dis<self.R:
                    e_ij=self.A+self.B*(math.pow(dis,self.V))
                    self.cost[i][j]=e_ij*self.nodes[i].e0_e1+self.e_receive*self.nodes[j].e0_e1

                else:
                    e_ij=1e50


        for i in range(self.num_nodes):
            dis = math.sqrt((self.nodes[i].x - self.s.x) ** 2 + (self.nodes[i].y - self.s.y) ** 2)
            if dis<self.R:
                shortest_dis[i]=(self.A+self.B*math.pow(dis,self.V))*self.nodes[i].e0_e1+self.e_receive

            parent[i]=self.num_nodes

        while True:
            n=-1
            cost_dis=1e50
            for i in range(self.num_nodes):
                if not used[i]:
                    if shortest_dis[i]<cost_dis:
                        n=i
                        cost_dis=shortest_dis[i]

            if n==-1:
                break
            used[n]=1
            for j in range(self.num_nodes):
                if not used[j] and self.cost[j][n]+cost_dis < shortest_dis[j]:
                    shortest_dis[j]=self.cost[j][n]+cost_dis
                    parent[j]=n

        for i in range(self.num_nodes):
            self.nodes[i].rece = 0
            self.nodes[i].send = 0

        for i in range(self.num_nodes):
            j=i
            while True:
                self.nodes[j].send+=self.nodes[i].sensor_rate
                if j!=i:
                    self.nodes[j].rece+=self.nodes[i].sensor_rate
                j=parent[j]

                if j>=self.num_nodes:
                    break

        for m in range(self.num_nodes):
            if parent[m]<self.num_nodes:
                dis=math.sqrt((self.nodes[m].x-self.nodes[parent[m]].x)**2+(self.nodes[m].y-self.nodes[parent[m]].y)**2)
                e_ij=self.A+self.B*math.pow(dis,self.V)
                self.nodes[m].c=e_ij*self.nodes[m].send+self.e_receive*self.nodes[m].rece
            elif parent[m]==self.num_nodes:
                dis = math.sqrt((self.nodes[m].x - self.s.x) ** 2 + (self.nodes[m].y - self.s.y) ** 2)
                e_ij=self.A+self.B*math.pow(dis,self.V)
                self.nodes[m].c=e_ij*self.nodes[m].send+self.e_receive*self.nodes[m].rece
            else:
                print('MST mistake')

    def step(self,action):
        self.s.x,self.s.y=self.sites[action].x,self.sites[action].y
        new_state = []
        if self.dyn:
            self.update_dyn_network(self.max_x,self.max_y)

        self.flow_routing()

        for j in range(self.num_sites):
            site_x, site_y = self.sites[j].x, self.sites[j].y
            new_state.append([self.mark_site,site_x/self.pos_normalization, site_y/self.pos_normalization,0,0])
        for i in range(self.num_nodes):
            self.nodes[i].e1-=self.nodes[i].c*self.DT
            self.nodes[i].e0_e1=self.nodes[i].e0/self.nodes[i].e1
            node_x,node_y=self.nodes[i].x, self.nodes[i].y
            new_state.append([self.mark_nodes,node_x/self.pos_normalization,node_y/self.pos_normalization,self.nodes[i].e1/self.nodes[i].e0,self.nodes[i].c/self.c_normalization])

        new_state[action][0]=self.mark_sink

        done=self.isdead()
        reward=0.1
        self.life_time +=1
        new_state = np.vstack((list(map(list,zip(*new_state))), self.matrix))

        return new_state,reward,done

    def get_a_size(self):
        return self.num_sites
    def get_s_size(self):
        return [self.state_size_w,self.state_size_l]
