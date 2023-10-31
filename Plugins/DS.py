import numpy as np 
from finquant import efficient_frontier
def efficient_frontier_weights(ef, min_return, max_return, len_targets=1250):
    #Compute the weights of efficient frontier    
    targets = np.linspace( np.round(min_return,8), np.round(max_return,8), len_targets)
    weights   = []
    for target in targets:
        x = ef.efficient_return(target, save_weights=False)
        weights.append(x)        
    X = np.array(weights)    
    return X

class DirectedSearch(): 
    def __init__(self,  profits, risk, eps, tol=1e-6, learn_rate=0.001, n_iter=500, freq=252): 
        #Optimal population
        #Problem parameters
        self.profits = profits 
        self.risk    = risk
        #Direction of search
        self.eps      = eps 
        #Learning rate
        self.learn_rate = learn_rate
        self.n_iter  = n_iter
        self.freq =freq
        self.tol = tol
        
    def evaluate_portf(self, x): 
        exp_return =  self.freq*(self.profits@x)
        exp_risk    = np.sqrt(self.freq)*np.sqrt(x@self.risk@x)
        return np.array([exp_risk, -exp_return])
    
    def jacob_portf(self, x): 
        D_risk   = np.sqrt(self.freq)* (self.risk@x)/np.sqrt((x@self.risk@x))
        D_return = self.freq*self.profits
        return np.vstack([D_risk, -D_return])
    
    def gradient_descent(self, x): 
        x_new = x 
        Fx_eps    = self.evaluate_portf(x) +self.eps
        for _ in range(self.n_iter): 
            J = self.jacob_portf(x_new)
            Jinv = np.linalg.pinv(J)
            candidate = x_new + self.learn_rate*(Jinv@self.eps)
            #Restricciones heurística de correción
            #candidate[candidate<0] = 0
            #candidate = candidate -np.min(candidate)
            #candidate = candidate/np.sum(candidate)
            if np.linalg.norm( Fx_eps -self.evaluate_portf(candidate),2) <self.tol: break
            else: x_new = candidate
        return x_new 
    
    def repair_solutions(self, X): 
        X[X < 0] = 0
        return X / X.sum(axis=1, keepdims=True)
    
    def directed_search(self, X):
        weights = []
        for x in X: 
            weights.append(self.gradient_descent(x))   
        return np.array(weights)
    
    def directed_search_full_space(self,X, steps): 
        weights = self.directed_search(X)
        full_sample = []
        for x, x_eps in zip(X, weights): 
            for t in np.linspace(0, 1, steps): 
                full_sample.append((1-t)*x + t*x_eps)
        return np.array(full_sample)
    def get_full_space_constrained(self, X, steps):
        X_full = self.directed_search_full_space(X,steps)
        return self.repair_solutions(X_full)
    
def get_Markowitz_directions(ef,PROFITS, RISK,min_return, max_return, eps,size=500, steps=50,tol=1e-12,learn_rate=0.001,  
                            n_iter=1000, freq=252):
    #Get the optimal results first 
    X_ef = efficient_frontier_weights(ef, min_return, max_return, size)
    ds = DirectedSearch(PROFITS, RISK, eps, tol, learn_rate, n_iter, freq)
    X_ef_new = ds.get_full_space_constrained(X_ef, steps)
    return X_ef_new[:]