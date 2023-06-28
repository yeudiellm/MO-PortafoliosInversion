import numpy as np 
import pandas as pd 
from tqdm import tqdm
from itertools import compress

def get_best_opt(data_pop, tol):
  #Copia de poblacion
  population = data_pop.copy()
  population = population.to_numpy()
  #Guardamos los índices 
  indx = data_pop.index
  #Archivo fantasma inicial 
  A = np.array( [[np.inf]*len(data_pop.columns)])
  best_idx = [None]
  #Iterar sobre los portafolios
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

def get_best_opt_eps(data_pop, tol, eps_array):
  #Castear epsilon
  eps_array = np.array(eps_array) 
  #Copia de la poblacion
  population = data_pop.copy()
  population = population.to_numpy()
  #Guardamos los índices 
  indx = data_pop.index
  #Archivo fantasma inicial
  A = np.array( [[np.inf]*len(data_pop.columns)])
  best_idx = [None]
  #Iterar sobre los portafolios 
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


#Esto hay que arreglarlo
def dist_H( A, vec): 
    L = np.linalg.norm( A-vec, ord=1, axis=1)
    return np.min(L)

def get_best_opt_eps_H(data_pop, tol, eps_array, delta):
  eps_array = np.array(eps_array)
  delta_array = np.array([delta]*len(data_pop.columns))
  population = data_pop.copy()
  population = population.to_numpy()
  indx = data_pop.index
  A = np.array( [[np.inf]*len(data_pop.columns)])
  best_idx = [None]
  
  for  idx, row in tqdm(zip(indx, population)): 
      test1 = (A +eps_array<= row).all(axis=1)
      test2 = np.linalg.norm(A+ eps_array-row, ord=1, axis=1) > tol 
      if not ((test1) & (test2)).any(): 
        if dist_H(A, row)>(delta/5):
          A = np.vstack([A,row])
          best_idx.append(idx)
          test1 = (row +eps_array +delta_array <= A).all(axis=1)
          test2 = np.linalg.norm( (row+eps_array+delta_array)- A, ord=1, axis=1) > tol
          A = A[~((test1) & (test2)) ,:]
          best_idx = list(compress(best_idx,~((test1) & (test2))))
  return pd.DataFrame(A, index=best_idx, columns=data_pop.columns)