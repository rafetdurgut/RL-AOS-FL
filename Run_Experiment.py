
#%% Run Experiment
# Her bir reward atamasından sonra kredi değerlerini tekrar hesapla.
# Reward sıfırdan küçükse de kredileri güncelle.
# Average ve min rewards 
import sys
from tkinter import W
from Experiment import Experiment
from Problem import *
from Operators import *
from BinaryABC import BinaryABC
from AOS import *

pNo = int(sys.argv[1])
rewardType = sys.argv[2]
W = int(sys.argv[3])
eps = float(sys.argv[4])
alpha = float(sys.argv[5])
gama = float(sys.argv[6])


problem = SetUnionKnapsack('Data/SUKP',pNo)
# problem = OneMax(100)
operator_pool = [  ibinABC(), disABC(), binABC()]
operator_selectors = ClusterRL(len(operator_pool), rewardType, W=W, alpha=alpha, gama=gama, Pmin=eps)
abc = BinaryABC(problem, pop_size=20, maxFE=40*max(problem.m, problem.n), limit=100)
alg_outs = ["convergence"]
aos_outs = ["credits","rewards","usage_counter","success_counter","cluster_history","credit_history","reward_history","operator_informations"]

exp = Experiment(abc,operator_pool,operator_selectors,
problem,algortihm_outs=alg_outs, aos_outs=aos_outs, runs=30, clear_history=True)
exp.Run()