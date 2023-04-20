from pymoo.core.problem import ElementwiseProblem

class Portfolio_Problem(ElementwiseProblem):
    def __init__(self,N, profits, risk):
        self.profits = profits
        self.risk = risk
        super().__init__(n_var=N,
                         n_obj=2,
                         #n_eq_constr=1,
                         xl=0,
                         xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        global portfolio_weights
        #NORMALIZAR X antes de empezar
        s = np.sum(x)
        x = x/s
        #Función de evaluacion
        profit_fitness = -self.profits@x
        risk_fitness    = np.sqrt(x@self.risk@x)
        out["F"] = [risk_fitness, profit_fitness]