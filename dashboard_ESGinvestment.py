#######################LIBRERÍAS###############################################
import streamlit as st 
#Básicos para manipulacion de datos 
import pandas as pd
import numpy as np
#Graficas 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
#sns.set_theme()
#Optimización multiobjetivo 
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.population import Population
#Finanzas
import yfinance as yf
import yesg
from finquant import efficient_frontier
#Plugins
from tqdm import tqdm
from itertools import compress
from Plugins import pre_processing
from Plugins import ArchievingStrategies
from Plugins import my_plotting
from Plugins import pymoo_extras

from pymoo.termination import get_termination
termination = get_termination("n_gen", 300)
from pymoo.algorithms.moo.nsga2 import NSGA2
nsgaii = NSGA2(pop_size=20,repair=pymoo_extras.Portfolio_Repair())
from pymoo.algorithms.moo.sms import SMSEMOA
smsemoa = SMSEMOA(pop_size=20, repair=pymoo_extras.Portfolio_Repair())

#######################CONFIGURACION PÁGINA###############################################
st.set_page_config(
    page_title = 'Optimización de Portafolios',
    page_icon = '✅',
    layout = 'wide'
)
st.title("Optimización de Portafolios utilizando ESG score")

colors = ['#7F7F7F', 'rgb(141,160,203)', 'rgb(27,158, 119)']
#######################PRE-PROCESSING###############################################
bursatil_dict = {'Dow and Jones': 'dowjones', 
                 'SP500': 'sp500', 
                 'Nasdaq100': 'nasdaq100',
                 'Otros': ''}
@st.cache_data
def get_assets(bursatil_index, assets_selection): 
    if bursatil_index=='Otros': 
        if len(assets_selection)>0: 
            assets = assets_selection
        else: 
            assets= ['ON', 'ORLY', 'PANW', 'VRTX', 'PEP', 'PCAR', 'AVGO', 'EXC', 'AAPL', 'GOOGL', 'META']
        
        ohlc = yf.download(assets, period="2y") 
        prices = ohlc["Adj Close"].dropna(how="all")
        returns, assets_info= pre_processing.get_assets_info(prices, 99, log_returns=True, drop_per_esg=True)
        best_assets = ArchievingStrategies.get_best_opt(assets_info.iloc[:, :2], 1e-6)
    else: 
        assets = pd.read_csv( 'Indices/tickers_'+str(bursatil_dict[bursatil_index])+'.csv', index_col=0)['0'].tolist()
        returns= pd.read_csv( 'data/'+str(bursatil_dict[bursatil_index])+'_returns.csv', index_col=0)
        assets_info=pd.read_csv( 'data/'+str(bursatil_dict[bursatil_index])+'_assets_info.csv', index_col=0)
        best_assets=pd.read_csv( 'data/'+str(bursatil_dict[bursatil_index])+'_best_assets.csv', index_col=0)
    return returns, assets_info, best_assets

#######################Haciendo Muestreo###############################################
@st.cache_data
def generating_sampling(algorithm_selection, eps): 
    if algorithm_selection=='Das-Dennis': 
        X = get_reference_directions("das-dennis", len(portfolio_problem.profits), n_partitions=10)
        F, ESG = pymoo_extras.eval_weights(portfolio_problem, X)
    elif algorithm_selection=='NSGA-II': 
        X, F, ESG =pymoo_extras.get_weights_with_pymoo(portfolio_problem, nsgaii, termination)
    elif algorithm_selection=='SMS-EMOA': 
        X, F, ESG =pymoo_extras.get_weights_with_pymoo(portfolio_problem, smsemoa, termination)

    FA =  pymoo_extras.annualised_portfolio_quantities(F)
    FA_best = ArchievingStrategies.get_best_opt(FA, 1e-6)
    FA_best_eps = ArchievingStrategies.get_best_opt_eps(FA, 1e-6, eps)
    FA_3D = FA_best_eps.copy()
    FA_3D['exp_esg'] = ESG[FA_best_eps.index]
    FA_3D_best       = ArchievingStrategies.get_best_opt(FA_3D, 1e-6)
    return FA, FA_best, FA_best_eps, FA_3D, FA_3D_best, X, ESG

placeholder = st.empty()

with placeholder.container():
    ratio_col1, ratio_col2, value_eps_col3, entry_col4 = st.columns(4)
    with value_eps_col3:
        return_deterioro=st.number_input('Deterioro en retorno', 
                        min_value=0.01, max_value=0.10, value=0.01)
        risk_deterioro = st.number_input('Deterioro en riesgo', 
                        min_value=0.01, max_value=0.10, value=0.01)
        
        eps = np.array([return_deterioro,risk_deterioro])
    with entry_col4: 
        assets_options = pd.read_csv( 'Indices/tickers_sp500.csv', index_col=0)['0'].tolist()
        assets_selection = st.multiselect('Seleccionar Activos', 
                                 assets_options, 
                                 ['ON', 'ORLY', 'PANW', 'VRTX', 'PEP', 'PCAR', 'AVGO', 'EXC', 'AAPL', 'GOOGL', 'META'])
    with ratio_col1: 
        bursatil_index = st.radio('índice Bursatil', options=bursatil_dict.keys(),index=0, key='burs_idx')
        returns, assets_info, best_assets = get_assets(bursatil_index, assets_selection)
        #######################Construyendo Portafolios###############################################
        PROFITS, RISK, ESG_SCORES = pre_processing.get_final_assets(returns[best_assets.index], assets_info.loc[best_assets.index])
        portfolio_problem = pymoo_extras.Portfolio_Problem(len(PROFITS), PROFITS, RISK, ESG_SCORES)
        #Construyendo Markowitz Portafolio
        ef = efficient_frontier.EfficientFrontier(pd.Series(PROFITS), pd.DataFrame(RISK), freq=252)
        ef_R = ef.efficient_frontier()
    with ratio_col2: 
        algorithm_selection = st.radio('Algoritmo de Muestreo', options= ['Das-Dennis', 'NSGA-II', 'SMS-EMOA'], index=0, key='alg')    
        FA, FA_best, FA_best_eps, FA_3D, FA_3D_best, X, ESG = generating_sampling(algorithm_selection, eps)
        
    
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1: 
        st.markdown("### Activos No Dominados")
        fig_assets = my_plotting.plot_assets_plotly(assets_info, best_assets)
        st.write(fig_assets)
        
    with fig_col2: 
        st.markdown("### Generando Muestra de Portafolios")
        frames = [FA, FA_best_eps, FA_best]
        labels = ['All', 'PQ-eps', 'PQ']
        fig_markowitz2D = my_plotting.plotting_samples_plotly(ef_R, frames, labels, colors)
        st.write(fig_markowitz2D)
    ## Correciones de signos 
    #En dos objetivos
    FA_best['exp_return'] = -FA_best['exp_return']
    FA_best['exp_esg'] =  ESG[FA_best.index] 
    #En tres objetivos
    FA_3D_best['exp_return'] = - FA_3D_best['exp_return']
    FA_3D['exp_return'] = - FA_3D['exp_return']
    FA_3D['Type'] = 'PQ-eps 2obj'
    FA_3D.loc[FA_3D_best.index, 'Type']='PQ 3obj'
    
    fig_col3, fig_col4 = st.columns(2)
    with fig_col3: 
        st.markdown("### Portafolios No Dominados en 3 objetivos")
        fig_markowitz3D = my_plotting.plotting_3D_plotly(FA_3D)
        st.write(fig_markowitz3D)
    with fig_col4: 
        st.markdown("### Proyección en 2 objetivos")
        fig_projection = my_plotting.plotting_projection_plotly(FA_3D_best, ef_R)
        st.write(fig_projection)
    
 
    st.markdown("### Histogramas de los objetivos")
    fig_projection = my_plotting.plot_histograms(FA_best, FA_3D_best)
    st.pyplot(fig_projection)
    
    st.markdown("### Proporción promedio de los activos")
    fig_proportions = my_plotting.plot_proportions(FA_best, FA_3D_best, X,best_assets, assets_info)
    st.pyplot(fig_proportions)