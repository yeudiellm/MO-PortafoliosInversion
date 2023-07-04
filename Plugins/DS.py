import numpy as np 

def efficient_frontier_weights(ef, len_targets=100, targets=None):
        min_return = ef.mean_returns.min() *ef.freq
        max_return = ef.mean_returns.max() *ef.freq
        targets = np.linspace( np.round(min_return,8), np.round(max_return,8), len_targets)
        # compute the efficient frontier
        efrontier = []
        weights   = []
        for target in targets:
            x = ef.efficient_return(target, save_weights=False)
            weights.append(x)        
        X = np.array(weights)    
        return X
    
class DirectedSearch(): 
    def __init__(self, X, profits, risk,d, learn_rate=0.005, freq=252, n_iter=50): 
        self.X = X
        self.profits = profits 
        self.risk    = risk
        self.d       = d
        self.learn_rate = learn_rate
        self.freq    = freq 
        self.n_iter  = n_iter
    
    def evaluate_portf(self, x): 
        exp_return =  self.profits@x
        exp_risk    = np.sqrt(x.T@self.risk@x)
        return np.array([exp_risk, -exp_return])
    
    def jacob_portf(self, x): 
        #anualizados
        D_risk   = np.sqrt(self.freq)* (self.risk@x)/np.sqrt((x@self.risk@x))
        D_return = -self.freq*self.profits
        return np.vstack([D_risk, D_return])
    
    def gradient_descent(self, x): 
        x_new = x 
        Fx    = self.evaluate_portf(x)
        for _ in range(self.n_iter): 
            J = self.jacob_portf(x_new)
            Jinv = np.linalg.pinv(J)
            #print(Jinv)
            candidate = x_new +self.learn_rate*(Jinv@self.d)
            #Restricciones heurística de correción
            candidate[candidate<0]=0
            s= np.sum(candidate)
            candidate  = candidate/s
        
            if np.any((Fx- self.evaluate_portf(candidate)) > self.d): break
            else: x_new = candidate
        return x_new 
    
    
    def directed_search(self):
        weights = []
        for x in self.X: 
            weights.append(self.gradient_descent(x))   
        return np.array(weights)
    
    def directed_search_full_space(self, steps): 
        weights = self.directed_search()
        full_sample = []
        for x, x_eps in zip(self.X, weights): 
            for t in np.linspace(0, 1, steps): 
                full_sample.append((1-t)*x + t*x_eps)
        return np.array(full_sample)