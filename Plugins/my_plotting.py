import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier,plotting

def plot_assets(assets_info, best_assets): 
    assets_plot = assets_info.copy()
    assets_plot['Activos'] = 'Dominado'
    assets_plot['Size']    = 1000
    assets_plot['exp_return'] = -assets_plot['exp_return']
    assets_plot.loc[best_assets.index,'Activos']='No-Dominado'
    assets_plot.loc[best_assets.index,'Size'] = 3000
    fig = plt.figure(figsize=(12,12))
    g= sns.scatterplot(data=assets_plot, x='exp_risk', y='exp_return', hue='Activos',
                    size='Size' ,palette=['cornflowerblue', 'red'])
    for idx, row in best_assets.iterrows(): 
        plt.text(row['exp_risk'], -row['exp_return'], idx, fontsize=12)
    h,l = g.get_legend_handles_labels()
    plt.legend(h[0:3],l[0:3], loc='best', fontsize=13)
    plt.xlabel('Riesgo esperado')
    plt.ylabel('Retorno Esperado')
    plt.show()
    
def plotting_samples(PROFITS, RISK, frames, labels, figsize): 
    ef = EfficientFrontier(PROFITS, RISK)
    fig, ax = plt.subplots(figsize=figsize)
    for i,f in enumerate(frames): 
        plt.scatter( f['exp_risk'], -f['exp_return'], label=labels[i])
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    plt.xlabel('exp_risk')
    plt.ylabel('exp_return')
    plt.legend()
    plt.show()