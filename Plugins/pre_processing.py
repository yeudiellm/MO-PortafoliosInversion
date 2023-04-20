#Se crea una función para verificar la completitud de las variables de la base.
import pandas as pd 
import yfinance as yf
import pypfopt
import numpy as np 
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
    #NOTA: Yahoo no esta scrapeando correctamente, hay que revisar. 
    sustainability = []
    for asset in assets: 
        dat_yf = yf.Ticker(asset)
        if dat_yf.sustainability is None:
            sustainability.append(100) 
        else:
            dat_yf = dat_yf.sustainability.to_dict()['Value']
            sustainability.append( dat_yf.get('totalEsg', 100))
    return np.array(sustainability)
        
def get_assets_info( prices,threshold, log_returns=True): 
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
    returns = pypfopt.expected_returns.returns_from_prices(prices, log_returns=log_returns)
    assets_ind_perf=pd.DataFrame()
    assets_ind_perf["Risk"]=returns.std()
    assets_ind_perf["Profit"]=-returns.mean()
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
    #esg_data = get_sustainability_data(returns.columns)
    return profits, risk