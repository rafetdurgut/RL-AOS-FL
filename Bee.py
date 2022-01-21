import numpy as np


class Bee:
    def __init__(self, problem=None, solution=None):
        self.problem = problem
        self.fitness = 0
        self.cost = 0
        self.prev_solution = []
        self.trial = 0
        self.features = []
        
        if solution is None:
            self.solution = np.random.random(self.problem.dimension) > 0.5
        else:
            self.solution = solution
        self.evaluate()

    def calculate_features(self, colony):
        #Solution Distance
        self.features = np.zeros((colony.feature_size,))
        self.features[0] = np.count_nonzero(colony.global_best.solution != self.solution)/self.problem.dimension

        #Objective Distance
        self.features[1] = (colony.global_best.cost - self.cost)/colony.global_best.cost

        #Trial Distance
        self.features[2] = (colony.landscape_features[-1][4] - self.trial)/100
        self.features[3] = (self.trial)/100

        #Updated Bits
        self.features[4] = np.count_nonzero(self.prev_solution != self.solution)/self.problem.dimension
        self.features[5] = (np.count_nonzero(colony.global_best.solution != self.prev_solution) - np.count_nonzero(colony.global_best.solution != self.solution))/self.problem.dimension
    
    def evaluate(self):
        self.solution, self.cost = self.problem.objective_function(self.solution)
        self.calculate_fitness()
        

    def initial(self):
        self.trial = 0
        self.solution = np.random.random(self.problem.dimension) > 0.5
        self.evaluate()

    def get_better(self, candidate):
        if candidate.cost > self.cost:
            candidate.trial = 0
            return candidate
        else:
            self.trial += 1
            return self

    def __str__(self):
        return f'Trial:{self.trial}, Cost:{self.cost}'

    def calculate_fitness(self):
        self.fitness = 1 + self.cost
