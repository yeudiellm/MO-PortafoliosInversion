from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

import numpy as np 
import pandas as pd
class Portfolio_Problem(ElementwiseProblem):
    def __init__(self,N, profits, risk,esg, **kwargs):
        self.profits = profits
        self.risk = risk
        self.esg = esg
        super().__init__(n_var=N,
                         n_obj=2,
                         xl=0.0,
                         xu=1.0,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        global portfolio_weights
        #NORMALIZAR X antes de empezar
        s = np.sum(x)
        x = x/s
        #Función de evaluacion
        exp_return = self.profits@x
        exp_risk    = np.sqrt(x.T@self.risk@x)
        exp_esg     = self.esg@x
        out["F"] = [exp_risk, -exp_return]
        out['ESG'] = exp_esg
        
class Portfolio_Repair(Repair):
    def _do(self, problem, X, **kwargs):
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)
    
def get_weights_with_pymoo(problem,algorithm,termination): 
    res = minimize(problem, 
                   algorithm, 
                   termination, 
                   save_history=True, 
                   verbose=False)
    all_pop = Population()
    for algorithm in res.history:
        all_pop = Population.merge(all_pop, algorithm.off)
    X = all_pop.get('X')
    F = all_pop.get('F')
    ESG = all_pop.get('ESG')
    pdF = pd.DataFrame(F, columns=['exp_risk', 'exp_return'])
    return X, pdF, ESG

def eval_weights(problem, weights): 
    Xpop = Population.new("X", weights)
    Xpop = Evaluator().eval(problem=problem, pop=Xpop)
    F = Xpop.get('F')
    ESG = Xpop.get('ESG')
    pdF = pd.DataFrame(F, columns=['exp_risk', 'exp_return'])
    return pdF, ESG