import pandas as pd 
import yfinance as yf
import yesg
import numpy as np 
from tqdm import tqdm
from finquant import returns as fq_returns

#We only considered historical data which a threshold of complete data. 
def completitud(df):
    """
    Check the fullness of data in a dataframe
    Args:
        df (pandas.DataFrame): Dataframe to analyze. 
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
    Function to obtain sustainability data. 
    If we don't have information we return np.nan for that stock. 
    Args:
        assets (list): List of strings with stocks symbols (Tickers)
    """
    sustainability = []
    for asset in assets:
        try: 
            esg_score = yesg.get_historic_esg(asset).tail(1)['Total-Score'][0] 
        except Exception as e: 
            esg_score = np.nan  
        sustainability.append(esg_score)            
    return np.array(sustainability)/100
        
def get_assets_info(prices,threshold, freq=252, log_returns=True, drop_per_esg=True): 
    """
    Obtaining basic information about stocks components.
    Dailiy returns, risk, esg information.
    Retornos diarios. Retorno esperado, Riesgo del activo. 
    Args:
        prices (pd.DataFrame): Historical prices of the stocks.
        threshold (float): Number in [0,1]. Percentage of NaN accepted values 
        log_returns (bool, optional): Si los retornos son logaritmicos.
        freq (int): Year of active trading
        log_returns (bool): If the returns are logarithmic or not
        drop_per_esg (bool): Delete stocks with no esg score information 
    Returns:
        returns (pd.DataFrame): Daily returns per stock.
        assets_ind_perf(pd.DataFrame) Mean-Return, Risk, Esg score per asset.
    """
    cc =completitud(prices)
    assets_up_thresh = cc[cc['completitud' ]>threshold]['variable'].values
    prices_selection = prices[assets_up_thresh]
    prices_selection = prices_selection.dropna(how='all')
    if log_returns: 
        returns = fq_returns.daily_log_returns(prices_selection)
    else:
        returns = fq_returns.daily_returns(prices_selection)
         
    assets_ind_perf = pd.DataFrame()
    #Annualised portfolio quantities
    assets_ind_perf["exp_risk"]=returns.std()*np.sqrt(freq)
    assets_ind_perf["exp_return"]=-returns.mean()*freq
    assets_ind_perf['esg_score'] = get_sustainability_data(prices_selection.columns)
    
    if drop_per_esg:
        assets_ind_perf = assets_ind_perf.dropna()
        returns = returns[assets_ind_perf.index]
    return returns, assets_ind_perf

def get_final_assets(returns, assets_ind_perf): 
    """
    Obtaining the objectives needed to the problem. 
    Mean Return, Covariance, ESG Score 
    Args: 
        returns (pd.DataFrame): Daily returns of assets. 
        assets_ind_perf (pd.DataFrame): Previos information computed about assets (esg scores)
    Returns:
        profits (np.array): Profits of each asset.
        cov  (np.array): Covariance of assets. 
        esg_data: ESG score for each profit. 
    """
    profits = returns.mean().to_numpy()
    risk     = returns.cov().to_numpy()
    esg_data = assets_ind_perf['esg_score'].to_numpy()
    return profits, risk, esg_data