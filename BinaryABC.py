import copy
from Bee import Bee
import numpy as np
from itertools import accumulate


class BinaryABC:
    def __init__(self, problem, pop_size, maxFE, limit, operator_selector= None, operator_pool=None, log_file=None):
        '''
        Constructor
        :param problem: The problem
        :param operator: Operator used by Bee's
        :param pop_size: Population size of bee's
        :param maxFE: Maximum number of evaluation size for objective function
        :param limit: Maximum number of trial for the bee.
        :param log_file: if none, nothing will be logged. Provide file name in order to write this file
        '''
        self.pop_size = pop_size
        self.problem = problem
        self.feature_size = 6

        self.maxFE = maxFE
        self.operator_pool = operator_pool
        self.operator_selector = operator_selector
        self.max_iteration =int(maxFE / (pop_size*2))
        self.limit = limit # bee trail limit
        self.colony = [Bee(problem) for _ in range(self.pop_size)]
        # keep best Bee
        self.global_best = Bee(problem)
        self.probabilities = np.zeros(self.pop_size)
        self.convergence = list()
        self.iteration = 0
        self.log_file = log_file
        self.FE = 0
        self.landscape_features = []
        
    def reset(self):
        self.colony = [Bee(self.problem) for _ in range(self.pop_size)]
        # keep best Bee
        self.global_best = Bee(self.problem)
        self.probabilities = np.zeros(self.pop_size)
        self.convergence = list()
        self.iteration = 0
        self.FE = 0
        self.landscape_features = []
        

    def employed_bee(self):
        self.calculate_cdf()
        for i in range(self.pop_size):
            candidate = copy.deepcopy(self.colony[i])
            neighbor = self.neighbor_selection()
            while self.colony[i] == neighbor:
                neighbor = self.neighbor_selection()
            op_no = self.operator_selector.operator_selection(candidate)
            self.FE += self.operator_pool[op_no].costFE()
            candidate = self.operator_pool[op_no].get_candidate(candidate, neighbor)
            self.operator_selector.add_reward(op_no, candidate, self.colony[i])
            self.colony[i] = self.colony[i].get_better(candidate)

    def onlooker_bee(self):
        self.calculate_cdf()
        max_fitness = max([bee.fitness for bee in self.colony])
        t = 0
        i = 0
        while t < self.pop_size:
            if np.random.random() < self.colony[i].fitness / max_fitness:
                t += 1
                candidate = copy.deepcopy(self.colony[i])
                neighbor = self.neighbor_selection()
                while self.colony[i] == neighbor:
                    neighbor = self.neighbor_selection()
                op_no = self.operator_selector.operator_selection(candidate)
                self.FE += self.operator_pool[op_no].costFE()
                candidate = self.operator_pool[op_no].get_candidate(candidate, neighbor)
                self.operator_selector.add_reward(op_no, candidate, self.colony[i])
                self.colony[i] = self.colony[i].get_better(candidate)
            i += 1
            i = i % self.pop_size

    def scout_bee(self):
        t_index=-1
        for ind, b in enumerate(self.colony):
            if b.trial >= self.limit:
                t_index = ind
                break
        if t_index > 0:
            self.colony[t_index].initial()

    def memorize(self):
        best_solution = max(self.colony, key=lambda b: b.cost)
        if best_solution.cost > self.global_best.cost:

            self.global_best = copy.deepcopy(best_solution)
            self.convergence.append((self.iteration, best_solution.cost))

            print(self.global_best.cost)
            



    def calculate_cdf(self):
        #CDF
        sum_fitness = sum([x.fitness for x in self.colony])
        self.probabilities = list(accumulate([x.fitness / sum_fitness for x in self.colony]))

    def neighbor_selection(self):
        r = np.random.random()
        n: int = next(index for index, i in enumerate(self.probabilities) if i > r)
        return self.colony[n]

    def stop_condition(self):
        # if one of the solution reach maxFE, returns true, else false
        if hasattr(self.problem, "best") and self.global_best.cost >= self.problem.best:
            return True
        return self.iteration >= self.max_iteration

    def derive_func_eval_count(self):
        return self.iteration * self.operator.costFE()

    def run(self):
        self.iteration = 0
        while not self.stop_condition():
            parents = copy.deepcopy(self.colony)
            best_solution = max(self.colony, key=lambda b: b.cost)
            parent_std = np.std([a.cost for  a in self.colony])
            parent_mean = np.mean([a.trial for  a in self.colony])
            best_known = best_solution.cost

            for b in self.colony:
                b.prev_solution = b.solution.copy()
                

            self.employed_bee()
            self.onlooker_bee()
    
            new_best_solution = max(self.colony, key=lambda b: b.cost)
            phi = new_best_solution.cost - best_known
            
            
            self.memorize()
            self.scout_bee()
            self.iteration += 1
            d = np.count_nonzero(self.global_best.solution != best_solution.solution) - np.count_nonzero(self.global_best.solution != new_best_solution.solution)
            Nstar = [ p for ind,p in enumerate(self.colony) if p.cost > parents[ind].cost]
            epp = len(Nstar)/self.pop_size
            
            if len(Nstar) == 0:
                eap = 0
            else:
                temp = 0
                for p in Nstar:
                    temp += (abs(best_known - p.cost)/self.pop_size)/parent_std
                eap = temp/ len(Nstar)
            self.landscape_features.append([phi, d, parent_std, epp*eap, parent_mean])
            for b in self.colony:
                b.calculate_features(self)
            self.operator_selector.next_iteration()
            

