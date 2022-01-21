import numpy as np
from os import listdir
from os.path import isfile, join
from copy import deepcopy


class AbstractBaseProblem:
    def objective_function(self):
        raise NotImplemented


class OneMax(AbstractBaseProblem):
    def __init__(self, dimension):
        self.pname = 'OneMax'
        self.dimension = dimension
        self.ID = str(dimension)
        self.best = dimension
        #Temp vars
        self.m = 100
        self.n=100

    def objective_function(self, solution):
        return solution, np.sum(solution==0)

class Task:
    def __init__(self, head, tail, s_cost, demand, dead_cost, inverse=-1):
        self.head = head
        self.tail=tail
        self.s_cost = s_cost
        self.demand=demand
        self.dead_cost=dead_cost
        self.inverse=inverse
        
class CARP(AbstractBaseProblem):
    MAX_NODE_NUM = 300
    def __init__(self, folderName, fileName):
        req_edge_num = nonreq_edge_num = 0
        self.tasks = []
        with open(f"DATA/CARP/{folderName}/{fileName}") as f:
            self.nov = int(f.readline().split()[0])
            self.noe = int(f.readline().split()[0])
            self.capacity = int(f.readline().split()[0])
            self.depot = int(f.readline().split()[0])
            
            self.trav_cost = np.full((self.nov+1, self.nov+1),np.inf)
            self.trav_cost[range(self.nov+1), range(self.nov+1)] = 0
            self.serv_cost = np.zeros((self.nov+1,self.nov+1))
            nonreq_edge_num = 0
            for i in range(self.nov):
                line = list(map(int, f.readline().split()))
                s_nod = line[0]
                for j in range(line[1]):
                    next_node = line[j*3 + 2]
                    self.trav_cost[s_nod][next_node] = line[j*3 + 3]
                    self.serv_cost[s_nod][next_node] = line[j*3 + 3]
                    if s_nod< next_node:
                        demand = line[j*3+4]
                        if demand > 0:
                            task_ = Task(s_nod, next_node, line[j*3 + 3], demand, line[j*3 + 3])
                            self.tasks.append(task_)
                        else:
                            nonreq_edge_num += 1
                            
            task_ = Task(self.depot, self.depot, 0, 0, 0, 0)
            req_edge_num = len(self.tasks)
            self.tasks.insert(0,task_)
            
            
            
            task_num = 2*req_edge_num
            
            
            for i in range(1, req_edge_num+1):
                self.tasks[i].inverse = i + req_edge_num
                
            for i in range(req_edge_num+1, 2*req_edge_num+1):
                print(i)
                task_ = Task(self.tasks[i-req_edge_num].tail, 
                             self.tasks[i-req_edge_num].head, 
                             self.tasks[i-req_edge_num].s_cost, 
                             self.tasks[i-req_edge_num].demand, 
                             self.tasks[i-req_edge_num].demand,
                             i-req_edge_num)
                self.tasks.append(task_)
                
            self.dijkstra()
            self.task_dist = np.zeros((task_num+1, task_num+1))
            for i in range(task_num+1):
                for j in range(task_num+1):
                    self.task_dist[i][j] = ((self.min_cost[self.tasks[i].head][self.tasks[j].head]) 
                    + (self.min_cost[self.tasks[i].head][self.tasks[j].tail])
                    + (self.min_cost[self.tasks[i].tail][self.tasks[j].head])
                    + (self.min_cost[self.tasks[i].tail][self.tasks[j].tail]))/4
                
            
                
            
            
            
    def dijkstra(self):
        sp = np.empty((self.noe+1, self.noe+1, self.noe+1), int)
        self.min_cost = np.full((self.noe, self.noe), np.inf)
        for i in range(1,self.noe+1):
            for j in range(1, self.noe+1):
                if i==j:
                    continue
                sp[i][j][0] = 1
                sp[i][j][1] = i
        mark = np.empty((self.MAX_NODE_NUM,), int)
        dist = np.empty((self.MAX_NODE_NUM,),float)
        dist1 = np.empty((self.MAX_NODE_NUM,),float)
        nearest_neighbor = np.empty((self.MAX_NODE_NUM,),int)
        
        for i in range(1, self.nov+1):
            mark[i] = 1
            for j in range(1,self.nov+1):
                if i==j:
                    continue
                
                mark[j] = 0
                dist[j] = self.trav_cost[i][j]
                dist1[j] = self.trav_cost[i][j]
            for k in range(1, self.nov+1):
                minimum = np.inf
                nearest_neighbor[0] = 0
                
                for j in range(1, self.nov+1):
                    if mark[j]:
                        continue
                    if dist1[j] == np.inf:
                        continue
                    if(dist1[j] < minimum):
                        minimum = dist1[j]
                if minimum == np.inf:
                    continue
                for j in range(1, self.nov+1):
                    if mark[j]:
                        continue
                    if dist1[j] == minimum:
                        nearest_neighbor[0]+=1
                        nearest_neighbor[nearest_neighbor[0]] = j
                v = nearest_neighbor[1]
                dist1[v] = np.inf
                mark[v] = 1
                
                if (sp[i][v][0] == 0) or (sp[i][v][0] > 0) and (sp[i][v][sp[i][v][0]] != v):    
                    sp[i][v][0] += 1
                    sp[i][v][sp[i][v][0]] = v
                
                for j in range(1,self.nov+1):
                    if mark[j]:
                        continue
                    
                    if (minimum + self.trav_cost[v][j]) <dist[j]:
                        dist[j] = minimum+self.trav_cost[v][j]
                        dist1[j] = minimum+self.trav_cost[v][j]
                    
                        for m in range(sp[i][v][0]+1):
                            sp[i][j][m] = sp[i][v][m]
                            
                for j in range(1, self.nov+1):
                    if i==j:
                        continue
                    self.min_cost[i][j] = dist[j]
            for i in range(1,self.nov+1):
                for j in range(1, self.nov+1):
                    if sp[i][j][0] == 1:
                        sp[i][j][0] == 0
                self.min_cost[i][i] = 0
                
            
                    
                
    def objective_function(self, solution):
        pass

class ZeroOneKnapsack(AbstractBaseProblem):
    def __init__(self, folderName, fileNo):
        mypath = folderName
        filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.dosyaAdi = filenames[fileNo]
        print(self.dosyaAdi)
        self.weights = []
        self.profits = []
        self.qualities = []
        self.non_qualities = []
        with open(f"{folderName}/{self.dosyaAdi}") as f:
            self.dimension, self.capacity = map(int, f.readline().split(' '))
            for i in range(self.dimension):
                line = f.readline().split(' ')
                self.weights.append(float(line[1]))
                self.profits.append(float(line[0]))
                self.qualities.append(float(line[0]) / float(line[1]))
                self.non_qualities.append(float(line[1]) / float(line[0]))

    'If the capacity is not fully filled'

    def optimizing_stage(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        qualities = np.multiply(self.qualities, solution == 0)
        add_index = np.argmax(np.multiply(qualities, solution==0))
        while cap_val + self.weights[add_index] <= self.capacity:
            solution[add_index] = True
            qualities[add_index] = 0
            cap_val += self.weights[add_index]
            add_index = np.argmax(qualities)
        return solution

    'If the capacity is exceed'

    def repair(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        qualities = np.multiply(self.non_qualities, solution)
        while cap_val > self.capacity:
            remove_index = np.argmax(qualities)
            solution[remove_index] = False
            qualities[remove_index] = 0
            cap_val -= self.weights[remove_index]
        return solution

    def objective_function(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        if cap_val > self.capacity:
            solution = self.repair(solution)
            solution = self.optimizing_stage(solution)
        else:
            solution = self.optimizing_stage(solution)

        cap_val = np.sum(self.weights, where=solution)
        #print("cap:"+ str(cap_val))
        sum_val = np.sum(self.profits, where=solution)
        return solution, sum_val

class SetUnionKnapsack(AbstractBaseProblem):
    def __init__(self, folderName, fileNo):
        mypath = folderName
        filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.ID = filenames[fileNo]
        self.dosyaAdi = filenames[fileNo]
        f = open("{}/{}".format(folderName, self.dosyaAdi), "r")
        print("{}/{}".format(folderName, filenames[fileNo]))
        f.readline()
        f.readline()
        line1 = f.readline()
        start = line1.index('=')
        stop = line1.index(' ', start)
        self.m = int(line1[start + 1:stop])
        self.dimension = self.m
        start = line1.index('=', stop)
        stop = line1.index(' ', start)
        self.n = int(line1[start + 1:stop])
        start = line1.index('=', stop)
        iseof = line1.find(' ', start)
        if iseof == -1:
            stop = len(line1) - 1
        else:
            stop = line1.index(' ', start)

        self.C = int(line1[start + 1:stop])

        f.readline()
        f.readline()
        self.p = list(map(int, f.readline().split()))

        f.readline()
        f.readline()
        self.w = list(map(int, f.readline().split()))

        f.readline()
        f.readline()
        self.rmatrix = np.zeros((self.m, self.n), dtype=bool)
        self.items = []
        for i in range(self.m):
            self.items.append([])
            rm = list(map(int, f.readline().split()))
            self.rmatrix[i, :] = deepcopy(rm[:])
            self.items[i] = np.where(self.rmatrix[i][:] == True)
        self.R = np.zeros(self.m)
        self.freqs = np.sum(self.rmatrix, axis=0)
        for i in range(self.m):
            self.R[i] = self.p[i] / np.sum(self.w[j] / self.freqs[j] for j in range(self.n) if self.rmatrix[i][j])

        self.H = np.argsort(self.R)[::-1][:self.m]
        f.close()

    def optimizing_stage(self, solution, temp):
        trial = temp.copy()
        for i in range(self.m):
            a = self.H[i]
            if solution[a] == 0:
                b = self.items[a]
                trial[b] = 1
                cap_val = np.sum(self.w, where=trial)
                if cap_val <= self.C:
                    temp[b] = True
                    solution[a] = True
                else:
                    trial = temp.copy()

        return solution

    def repair(self, solution):
        temp_sol = np.zeros(self.m, dtype=bool)
        temp = np.zeros((self.n), dtype=bool)
        trial = np.array(temp, copy=True)

        for i in range(self.m):
            a = self.H[i]
            if solution[a]:
                b = self.items[a]
                trial[b] = True
                cap_val = np.sum(self.w, where=trial)
                if cap_val <= self.C:
                    temp[b] = True
                    temp_sol[a] = True
                else:
                    trial = temp.copy()
        return temp_sol, temp

    def objective_function(self, solution):
        temp = np.zeros((self.n), dtype=bool)
        for i in range(self.m):
            if solution[i]:
                temp[self.items[i]] = True
        cap_val = np.sum(self.w, where=temp)

        if cap_val > self.C:
            solution, temp = self.repair(solution)
            solution = self.optimizing_stage(solution, temp)
        else:
            solution = self.optimizing_stage(solution, temp)

        sum_val = np.sum(self.p[i] for i, val in enumerate(solution) if val)
        return solution, sum_val
