import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def plot_assets(assets_info, best_assets): 
    assets_plot = assets_info.copy()
    assets_plot['Activos'] = 'Dominado'
    assets_plot['Size']    = 4000
    assets_plot['exp_return'] = -assets_plot['exp_return']
    assets_plot.loc[best_assets.index,'Activos']='No-Dominado'
    assets_plot.loc[best_assets.index,'Size'] = 5000
    #fig = plt.figure(figsize=(12,12))
    g= sns.scatterplot(data=assets_plot, x='exp_risk', y='exp_return', hue='Activos',
                    size='Size' ,palette={'Dominado':'cornflowerblue','No-Dominado':'red'})
    for idx, row in best_assets.iterrows(): 
        plt.text(row['exp_risk']+0.005, -row['exp_return']+0.005, idx, fontsize=12)
    h,l = g.get_legend_handles_labels()
    plt.legend(h[0:3],l[0:3], loc='best', fontsize=13)
    plt.xlabel('Riesgo anual esperado')
    plt.ylabel('Retorno anual Esperado')
    plt.show()
    
def plotting_samples(ef_R, frames, labels, figsize):
    fig = plt.figure(figsize=figsize)
    plt.plot(ef_R[:,0], ef_R[:, 1], linestyle='-', color='black', lw=1, label='Markowitz') 
    
    for i,f in enumerate(frames): 
        plt.scatter( f['exp_risk'], -f['exp_return'], label=labels[i])
    plt.xlabel('Riesgo anual esperado')
    plt.ylabel('Retorno anual Esperado')
    plt.legend()
    plt.show()
    
def plot_assets_plotly(assets_info, best_assets): 
    assets_plot = assets_info.copy()
    assets_plot['Stocks'] = 'Dominado'
    assets_plot['Size']    = 2000
    assets_plot['exp_return'] = -assets_plot['exp_return']
    assets_plot.loc[best_assets.index,'Stocks']='No-Dominado'
    assets_plot.loc[best_assets.index,'Size'] = 5000
    fig = px.scatter( assets_plot, x='exp_risk', y='exp_return', text=assets_plot.index,
                      color="Stocks", size='Size',
                      width=800, height=600
                     )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title = "Riesgo Anual Esperado",
        yaxis_title = "Retorno Anual Esperado",
    )
    fig.show()
    return fig
    
def plotting_samples_plotly(ef_R, frames, labels, colors):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=ef_R[:,0], y=ef_R[:,1],
                    mode='lines',
                    name='Markowitz', 
                    marker_color='#222A2A'), 
                 )

    for i,f in enumerate(frames): 
        fig.add_trace(go.Scatter(x=f['exp_risk'], y=-f['exp_return'], 
                             mode='markers', 
                             name=labels[i],
                             marker_color=colors[i]),
                     )
    fig.update_layout(
        xaxis_title = "Riesgo Anual Esperado",
        yaxis_title = "Retorno Anual Esperado",
    )
    fig.show()
    return fig