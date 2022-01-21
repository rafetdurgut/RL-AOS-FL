import numpy as np

#Learning mode : 0 ; Starts from zero for each runtime.
#Learning mode : 1 ; Learn from first runtime and then freeze.
#Learning mode : 2 ; Continuously learn for all runtime.

class abstractOperatorSelection:
    def __init__(self, operator_size, reward_type, W=5, alpha=0.1, beta=0.5, Pmin=0.1, learning_mode=0):
        self.learning_mode = learning_mode
        self.operator_size = operator_size
        self.rewards = [[0] for _ in range(self.operator_size)]
        self.rewards_history =  np.zeros((operator_size))
        self.credits = [[0] for _ in range(self.operator_size)]
        self.credits_history = [[] for _ in range(self.operator_size)]
        self.success_counter = [[0] for _ in range(operator_size)]
        self.total_succ_counters = np.zeros((operator_size))
        self.usage_counter = [[0] for _ in range(operator_size)]
        self.probabilities = np.zeros((operator_size))
        self.reward = np.zeros((operator_size))
        self.type = 'iteration'
        self.iteration = 0
        self.reward_type = reward_type
        self.W = W
        self.Pmin = Pmin
        self.Pmax = 1 - (self.operator_size - 1) * Pmin
        self.alpha = alpha
        self.beta = beta
        self.operator_informations = []
        self.runtime = 0

    def reset(self):
        self.rewards = [[0] for _ in range(self.operator_size)]
        self.rewards_history =  np.zeros((self.operator_size))
        self.credits = [[0] for _ in range(self.operator_size)]
        self.credits_history = [[] for _ in range(self.operator_size)]
        self.success_counter = [[0] for _ in range(self.operator_size)]
        self.total_succ_counters = np.zeros((self.operator_size))
        self.usage_counter = [[0] for _ in range(self.operator_size)]
        self.probabilities = np.zeros((self.operator_size))
        self.reward = np.zeros((self.operator_size))
        self.Pmax = 1 - (self.operator_size - 1) * self.Pmin
        self.iteration = 0
        self.runtime += 1
        self.operator_informations = []
        self.timer = np.zeros((self.operator_size),dtype=int)
        self.cluster_history = [[] for _ in range(self.operator_size)]
        self.n_cluster_history = [[] for _ in range(self.operator_size)]


    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        if isinstance(self, ClusterRL):
            self.create_clusters()

    def get_reward(self, new_fitness, old_fitness):
        r = (self.algorithm.problem.dimension / self.algorithm.global_best.cost) * float((new_fitness - old_fitness))
        # Add reward to rewards
        # Update Credits ...
        # if r < 0:
        #     r = 0
        return r

    def next_iteration(self):
        self.iteration += 1
        for i in range(self.operator_size):
            self.credits_history[i].append(self.credits[i][-1])
            self.operator_informations.append([i, self.runtime, self.iteration, self.credits_history[i][-1], 
            self.rewards_history[i], self.usage_counter[i][-1], self.success_counter[i][-1]])
            # self.rewards[i].append(0)
            self.usage_counter[i].append(0)
            self.rewards_history[i] = 0
            self.success_counter[i].append(0)

    def add_reward(self, op_no, candidate, current):
        self.usage_counter[op_no][self.iteration] += 1
        reward = self.get_reward(candidate.cost, current.cost)
        if reward > 0:
            self.success_counter[op_no][self.iteration] += 1
            self.total_succ_counters[op_no] += 1


    def apply_rewards(self, i):
        if self.reward_type == "insta":
            reward = self.rewards[i][self.iteration]
        elif self.reward_type == "average":
            start_pos = max(0, len(self.rewards[i]) - self.W)
            reward = np.average(self.rewards[i][start_pos:len(self.rewards[i])])
        elif self.reward_type == "extreme":
            start_pos = max(0, len(self.rewards[i]) - self.W)
            reward = np.max(self.rewards[i][start_pos:len(self.rewards[i])])
        return reward

    def update_credits(self,op_no):
        r = self.apply_rewards(op_no)
        credit = (1 - self.alpha) * self.credits[op_no][-1] + self.alpha * r
        self.credits[op_no].append(credit)

    def operator_selection(self, candidate=None):
        raise Exception("Should not call Abstract Class!")

    def roulette_wheel(self, ):
        sumProbs = sum(self.probabilities)
        probs = [item / sumProbs for item in self.probabilities]
        op = np.random.choice(len(probs), p=probs)
        return op


class ClusterRL(abstractOperatorSelection):
    def __init__(self, operator_size, reward_type, W, alpha,  Pmin, gama = 0.3,learning_mode=0):
        super(ClusterRL, self).__init__(operator_size, reward_type, W, alpha=alpha, beta=0.1, Pmin=Pmin,learning_mode=learning_mode)
        self.operator_size = operator_size
        self.learning_mode = learning_mode
        self.alpha = alpha
        self.type = 'function'
        self.gama = gama
        self.timer = np.zeros((self.operator_size),dtype=int)
        self.cluster_history = [[] for _ in range(self.operator_size)]
        self.n_cluster_history = [[] for _ in range(self.operator_size)]
        
        
    def create_clusters(self):
        self.clusters = np.zeros((self.operator_size, self.algorithm.feature_size))

    def get_reward_bytype(self,op_no):
        if self.timer[op_no] == 0:
            return self.rewards[op_no][-1]
        if self.reward_type == "insta":
            return self.rewards[op_no][-1]
        elif self.reward_type == "average":
            start_pos = max(0, len(self.rewards[op_no]) - self.W)
            reward = np.sum(self.rewards[op_no][start_pos:-1])/(self.iteration-start_pos)
            return  reward
        elif self.reward_type == "extreme":
            start_pos = max(0, len(self.rewards[op_no]) - self.W)
            reward = np.max(self.rewards[op_no][start_pos:-1])
            return reward

    def add_reward(self, op_no, candidate, current):
        super(ClusterRL, self).add_reward(op_no, candidate, current)
        reward = self.get_reward(candidate.cost, current.cost)
        self.rewards_history[op_no] += reward
        self.rewards[op_no].append(reward)
        self.update_credits(op_no)
        # r = reward + self.gama * self.distance(op_no, current)
        if reward > 0:
            self.update_cluster(op_no, current)
            self.timer[op_no] += 1
        
        #self.rewards[op_no].append(reward)
            
            #self.iter_rewards[op_no][self.iteration] += reward
            #self.iter_credits[op_no][self.iteration] += credit
            #self.credits[op_no].append(credit)
            #self.iter_rewards[op_no][self.iteration] += r + self.gama * self.hamming_distance(self.clusters[op_no], candidate.solution)

    def update_cluster(self, op, current):
        
        if len(current.features) == 0:
            return
        
        if np.sum(self.clusters[op][:]) == 0 :
            for i in range( len(current.features) ) :
                self.clusters[op][i] = current.features[i]
            return
        
        for i in range(len(current.features) ) :
            self.clusters[op][i] = self.clusters[op][i] + (current.features[i]-self.clusters[op][i]) / (
                        self.timer[op] + 1)
        
        self.cluster_history.append(np.concatenate(([op, self.iteration, self.runtime], self.clusters[op])))

    
        
    def distance(self, op, candidate):
        dist = 0
        if self.iteration==0 or len(candidate.features)==0:
            return dist
        return  (self.algorithm.problem.dimension/self.algorithm.feature_size)*np.linalg.norm(self.clusters[op] - candidate.features)
        
        
    def operator_selection(self, candidate):
        if (np.all(self.total_succ_counters) == 0) or np.random.rand() < self.Pmin:
            for i in range(self.operator_size):
                self.probabilities[i] = 1
            return self.roulette_wheel()
        values = [-1 * self.credits[ind][-1] + self.gama * self.distance(ind, candidate) for ind in range(self.operator_size)]
        best_op = np.argmin(values)
        return best_op
        return op

    def __conf__(self):
        return ['CLRL', self.operator_size,  self.reward_type, self.Pmin, self.W, self.alpha, self.gama,self.learning_mode]

