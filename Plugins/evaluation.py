from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import pandas as pd
class Evaluator_Solutions(): 
    def __init__(self, bursatil_index:str,directory: str, methods:list, n_ejec:int, n_obj:int):
        #Init methods
        self.bursatil_index = bursatil_index
        self.directory = directory
        self.methods = methods
        self.n_ejec = n_ejec
        self.n_obj  = n_obj
        self.ef_R = np.load(directory+'ef_R.npy')
        self.ef_R[:, 1] = -self.ef_R[:, 1] 
        self.ref_point  = np.array([1.1]*n_obj)
        
        GD_ind = GD(self.ef_R)
        IGD_ind = IGD(self.ef_R)
        GDplus_ind = GDPlus(self.ef_R)
        IGDplus_ind = IGDPlus(self.ef_R) 
        HV_ind  =HV(ref_point=np.array([1.1]*n_obj))
        self.inds_names = ['GD', 'IGD', 'GDplus', 'IGDplus', 'HV', '|PQeps|']
        self.inds  = [GD_ind, IGD_ind,GDplus_ind,IGDplus_ind, HV_ind]
        
    def get_max_min(self): 
        #20 ejecuciones, 5 operadores, 100 ultimas generaciones 
        max_matrix     = np.empty(shape=(self.n_ejec*len(self.methods), self.n_obj))
        min_matrix     = np.empty(shape=(self.n_ejec*len(self.methods), self.n_obj))
        for j,method in enumerate(self.methods): 
            for i in range(self.n_ejec): 
                FA_eps= pd.read_csv(self.directory +'FA_'+method+'_eps'+str(i)+'.csv', index_col=0)
                FA_eps_numpy = FA_eps.to_numpy()
                max_matrix[j*self.n_ejec+i,:] = np.max(FA_eps_numpy, axis=0)
                min_matrix[j*self.n_ejec+i,:] = np.min(FA_eps_numpy, axis=0)   
        general_max = np.max(max_matrix, axis=0)
        general_min = np.min(min_matrix, axis=0)
        return general_max, general_min
    
    def minmaxScaler(self,X, general_max, general_min, new_max, new_min): 
        X_std = (X -general_min)/(general_max-general_min)
        X_scaled = X_std*(new_max-new_min) + new_min
        return X_scaled
 
    def get_opt_ind(self, name_ind, ind): 
        values_ind = np.empty(shape=(len(self.methods), self.n_ejec))
        if name_ind=='HV':
            general_max, general_min = self.get_max_min() 
        for j, method in enumerate(self.methods): 
            for i in range(self.n_ejec): 
                FA_eps= pd.read_csv(self.directory +'FA_'+method+'_eps'+str(i)+'.csv', index_col=0)
                FA_eps_numpy = FA_eps.to_numpy()
                if name_ind=='HV':
                    FA_eps_norm = self.minmaxScaler(FA_eps_numpy, general_max, general_min, 1, 0)
                    values_ind[j, i] = ind(FA_eps_norm)
                elif name_ind=='|PQeps|':
                    values_ind[j, i] = len(FA_eps_numpy)
                else: 
                    values_ind[j,i] = ind(FA_eps_numpy)
        return values_ind
                        
    def get_final_reports(self, save_file=False):
        S = pd.DataFrame()
        for name_ind, ind in zip(self.inds_names, self.inds): 
            print(name_ind)
            values_ind =self.get_opt_ind(name_ind, ind)
            for j, method in enumerate(self.methods): 
                result = {'BursatilIndex': self.bursatil_index, 
                          'Sampling Method': method, 
                          'Indicator': name_ind, 
                          'Execution': range(self.n_ejec), 
                          'Ind_Value': values_ind[j]
                          }
                result = pd.DataFrame(result)
                S = pd.concat([S, result], ignore_index=True)
        if save_file: 
            S.to_csv('FinalResults/'+self.bursatil_index +'.csv', index=False)
        return S    