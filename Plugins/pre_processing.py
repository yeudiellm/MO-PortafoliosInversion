#Se crea una función para verificar la completitud de las variables de la base.
import pandas as pd 
import yfinance as yf
import yesg
import pypfopt
import numpy as np 
from tqdm import tqdm
from finquant import returns as fq_returns

def completitud(df):
    """
    Revisa la completitud de un dataframe
    Args:
        df (pandas.DataFrame): Dataframe a examinar. 
    Returns:
        comple: DataFrame con la completitud por columna del dataset original.
    """
    comple=pd.DataFrame(df.isnull().sum())
    comple.reset_index(inplace=True)
    comple=comple.rename(columns={"index":"variable",0:"total"})
    comple["completitud"]=(1-comple["total"]/df.shape[0])*100
    comple=comple.sort_values(by="completitud",ascending=True)
    comple.reset_index(drop=True,inplace=True)
    return comple

def get_sustainability_data(assets): 
    """
    Función para obtener la información medio-ambiental. 
    Si no se reporta la información medio-ambiental se le da la peor nota. 

    Args:
        assets (list): Lista de strings (nombres de los activos)
    """
    sustainability = []
    for asset in tqdm(assets):
        try: 
            esg_score = float(yesg.get_esg_short(asset)['Total-Score'][0])
        except: 
            esg_score = np.nan #Las que no tienen informacion se asigna la peor tasa posible. 
        sustainability.append(esg_score)            
    return np.array(sustainability)/100
        
def get_assets_info(prices,threshold, freq=252, log_returns=True, drop_per_esg=True): 
    """
    Función que obtiene la información básica del portafolio. 
    Retornos diarios. Retorno esperado, Riesgo del activo. 
    Matriz de covarianzas.

    Args:
        prices (pandas.DataFrame): Dataset historico del precio de los activos.
        threshold (float): Number in [0,1]. Cantidad de observaciones mínimas para 
                           considerar el activo. 
        log_returns (bool, optional): Si los retornos son logaritmicos.

    Returns:
        _type_: _description_
    """
    cc =completitud(prices)
    assets_up_thresh = cc[cc['completitud' ]>threshold]['variable'].values
    prices_selection = prices[assets_up_thresh]
    if log_returns: 
        returns = fq_returns.daily_log_returns(prices_selection)
    else:
        returns = fq_returns.daily_returns(prices_selection)
         
    assets_ind_perf = pd.DataFrame()
    assets_ind_perf["exp_risk"]=returns.std()*np.sqrt(freq)
    assets_ind_perf["exp_return"]=-returns.mean()*freq
    assets_ind_perf['esg_score'] = get_sustainability_data(prices_selection.columns)
    
    if drop_per_esg:
        assets_ind_perf = assets_ind_perf.dropna()
        returns = returns[assets_ind_perf.index]
    return returns, assets_ind_perf

def get_final_assets(returns): 
    """
    Obtiene los promedios y la matriz de covarianzas 
    de las acciones a considerar para el portafolio.

    Args:
        returns (pandas.DataFrame): _description_

    Returns:
        profits, risk (np.ndarra): ndarray con los promedios y matrices de covarianza.
    """
    profits = returns.mean().to_numpy()
    risk     = returns.cov().to_numpy()
    esg_data = get_sustainability_data(returns.columns)
    return profits, risk, esg_data