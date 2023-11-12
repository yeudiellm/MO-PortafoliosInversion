import numpy as np
import pandas as pd
def rand_weights(n_portfolios, n_assets):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.random((n_portfolios, n_assets))
    return k / np.sum(k, axis=1, keepdims=True)

def random_returns(n_assets,n_obs=504,  freq=252):
    ''' Produces n assets with random normal returns'''
    seed = np.random.randint(0, 10000) 
    rng = np.random.default_rng(seed)
    returns_vec = rng.normal(loc=0, scale=0.01, size=(n_obs, n_assets))
    while np.any(0 > np.mean(returns_vec, axis=0)): 
        if np.sum(np.mean(returns_vec))*freq >0.5: 
            seed = np.random.randint(0, 10000)
            rng= np.random.default_rng(seed)
            returns_vec = rng.normal(loc=0, scale=0.01, size=(n_obs, n_assets))
            continue
        seed = np.random.randint(0, 10000)
        rng= np.random.default_rng(seed)
        returns_vec = rng.normal(loc=0, scale=0.01, size=(n_obs, n_assets))
    returns =pd.DataFrame(returns_vec, columns = ['s'+str(i+1) for i in range(n_assets)])
    assets_info = pd.DataFrame()
    assets_info["exp_risk"]=returns.std()*np.sqrt(freq)
    assets_info["exp_return"]=-returns.mean()*freq
    assets_info["esg_score"] = rng.uniform(low=0.0, high=1.0, size=(n_assets))
    print("The seed was: ", seed)        
    return seed, returns, assets_info