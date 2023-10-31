import numpy as np 
import pandas as pd 
from tqdm import tqdm
from itertools import compress

def get_best_opt(data_pop, tol):
  """Obtaining the best portfolios (stocks) according to objectives
  Args:
      data_pop (pd.DataFrame): DataFrame with objectives information per portfolio (stock)
      tol (float): tolerance to accept different solutions
  Returns:
      (pd.DataFrame) : DataFrame with no dominated solutions
  """
  #Saving population, conver to numpy
  population = data_pop.copy()
  population = population.to_numpy()
  #Keep index 
  indx = data_pop.index
  #Init file 
  A = np.array( [[np.inf]*len(data_pop.columns)])
  best_idx = [None]
  #Review portfolios
  for idx,row in tqdm(zip(indx,population)):
    test1 = (A <= row).all(axis=1)
    test2 = np.linalg.norm(A-row, ord=2, axis=1) > tol
    if not ((test1) & (test2)).any(): 
      A = np.vstack([A,row])
      best_idx.append(idx)
      test1 = (row <= A).all(axis=1)
      test2 = np.linalg.norm(row- A, ord=2, axis=1)> tol
      A = A[~((test1) & (test2)) ,:]
      best_idx = list(compress(best_idx,~((test1) & (test2))))
  return pd.DataFrame(A, index=best_idx, columns=data_pop.columns)

def get_best_opt_eps(data_pop, tol, eps_array):
  """Obtaining the best portfolios (stocks) according to objectives
  Args:
      data_pop (pd.DataFrame): DataFrame with objectives information per portfolio (stock)
      tol (float): tolerance to accept different solutions
      eps_array (np.array): epsilon tolerance for each objective. 
  Returns:
      (pd.DataFrame) : DataFrame with (eps) no dominated solutions
  """
  #Casting epsilon
  eps_array = np.array(eps_array) 
  #Saving population, conver to numpy
  population = data_pop.copy()
  population = population.to_numpy()
  #Keep index  
  indx = data_pop.index
  #Init file 
  A = np.array( [[np.inf]*len(data_pop.columns)])
  best_idx = [None]
  #Review portfolios 
  for  idx, row in tqdm(zip(indx, population)):
    test1 = (A +eps_array<= row).all(axis=1)
    test2 = (np.linalg.norm( (A+ eps_array)-row, ord=1, axis=1) > tol) 
    if not ((test1) & (test2)).any(): 
      A = np.vstack([A,row])
      best_idx.append(idx)
      test1 = (row +eps_array<= A).all(axis=1)
      test2 = np.linalg.norm( (row+eps_array)- A, ord=1, axis=1) > tol
      A = A[~((test1) & (test2)) ,:]
      best_idx = list(compress(best_idx,~((test1) & (test2))))
  return pd.DataFrame(A, index=best_idx, columns=data_pop.columns)


