import numpy as np 
import pandas as pd 
from tqdm import tqdm
from itertools import compress

def get_best_opt(data_pop, tol):
  population = data_pop.copy()
  population = population.to_numpy()
  indx = data_pop.index
  A = np.array( [[np.inf]*len(data_pop.columns)])
  best_idx = [np.inf]
  for idx,row in tqdm(zip(indx,population)):
    test1 = (A <= row).all(axis=1)
    test2 = np.linalg.norm(A-row, ord=1, axis=1) > tol
    if not ((test1) & (test2)).any(): 
      A = np.vstack([A,row])
      best_idx.append(idx)
      test1 = (row <= A).all(axis=1)
      test2 = np.linalg.norm(row- A, ord=1, axis=1)> tol
      A = A[~((test1) & (test2)) ,:]
      best_idx = list(compress(best_idx,~((test1) & (test2))))
  return pd.DataFrame(A, index=best_idx, columns=data_pop.columns)