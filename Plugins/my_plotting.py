import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import numpy as np 
sns.set_theme()
#########################################################################################################################################
#MATPLOTLIB GENERIC
def plot_assets(assets_info, best_assets, figsize): 
    fig = plt.figure(figsize=figsize)
    assets_plot = assets_info.copy()
    assets_plot['Stocks'] = 'Dominated'
    assets_plot['Size']    = 4000
    assets_plot['exp_return'] = -assets_plot['exp_return']
    assets_plot.loc[best_assets.index,'Stocks']='Not-Dominated'
    assets_plot.loc[best_assets.index,'Size'] = 5000
    g= sns.scatterplot(data=assets_plot, x='exp_risk', y='exp_return', hue='Stocks',
                    size='Size' ,palette={'Dominated':'cornflowerblue','Not-Dominated':'red'})
    for idx, row in best_assets.iterrows(): 
        plt.text(row['exp_risk']+0.005, -row['exp_return']+0.005, idx, fontsize=12)
    h,l = g.get_legend_handles_labels()
    plt.legend(h[0:3],l[0:3], loc='best', fontsize=13)
    plt.xlabel("Expected Annual Risk")
    plt.ylabel("Expected Annual Return")
    plt.show()
    return fig
def plot_assets_with_ax(assets_info, best_assets, ax): 
    ax.scatter( assets_info['exp_risk'], -assets_info['exp_return'], label=r'$Q$', color='cornflowerblue', s=2)
    ax.scatter( best_assets['exp_risk'], -best_assets['exp_return'], label=r'$P_{Q}$', color='red', s=8)
    ax.legend(title='Assets', loc='lower right', fontsize='9')
    ax.set_xlabel("Expected Annual Risk", fontsize='9')
    ax.set_ylabel("Expected Annual Return", fontsize='9')
    return ax
    
def plotting_samples_with_ax(ef_R, frames, labels,colors, ax):
    ax.plot(ef_R[:,0], ef_R[:, 1], linestyle='dashed', color='black', lw=2, label='Markowitz') 
    for i,f in enumerate(frames): 
        ax.scatter( f['exp_risk'], -f['exp_return'], label=labels[i], color=colors[i], s=2)
    ax.set_xlabel("Expected Annual Risk", fontsize='9')
    ax.set_ylabel("Expected Annual Return", fontsize='9')
    ax.legend(loc='lower right')
    return ax

def plotting_projection_with_ax(FA_3D_best, ef_R, colormap, vmin, vmax, ax): 
    ax.plot(ef_R[:,0], ef_R[:, 1], linestyle='dashed', color='black', lw=2, label='Markowitz')
    points =ax.scatter( FA_3D_best['exp_risk'], 
                        -FA_3D_best['exp_return'], 
                        s=2,
                        c=FA_3D_best['exp_esg'], cmap=colormap, vmin=vmin, vmax=vmax)
    plt.colorbar(points,ax=ax, label='ESG risk score')
    ax.set_xlabel("Expected Annual Risk", fontsize='9')
    ax.set_ylabel("Expected Annual Return", fontsize='9')
    ax.legend(loc='lower right')
    return ax 
    
#########################################################################################################################################
#PLOTLY FRAMES     
def plot_assets_plotly(assets_info, best_assets, with_labels=True): 
    assets_plot = assets_info.copy()
    assets_plot['Stocks'] = 'Dominated'
    assets_plot['Size']    = 2000
    assets_plot['exp_return'] = -assets_plot['exp_return']
    assets_plot.loc[best_assets.index,'Stocks']='Not-Dominated'
    assets_plot.loc[best_assets.index,'Size'] = 5000
    assets_plot = assets_plot.sort_values(by='Stocks')
    if with_labels: 
        fig = px.scatter( assets_plot, x='exp_risk', y='exp_return', text=assets_plot.index,
                      color_discrete_sequence=['#636EFA','#EF553B'],
                      color="Stocks", size='Size',
                      #width=800, height=600
                     )
    else: 
        fig = px.scatter( assets_plot, x='exp_risk', y='exp_return',
                      color_discrete_sequence=['#636EFA','#EF553B'],
                      color="Stocks", size='Size',
                      #width=800, height=600
                     )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title = "Expected Annual Risk",
        yaxis_title = "Expected Annual Return",
    )
    #fig.show()
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
        xaxis_title = "Expected Annual Risk",
        yaxis_title = "Expected Annual Return",
    )
    #fig.show()
    return fig

def plotting_3D_plotly(FA_3D):
    fig = px.scatter_3d(FA_3D.sort_values(by='Type'), x='exp_risk', y='exp_return', z='exp_esg',
                    color_discrete_sequence=['rgb(27,158, 119)','rgb(141,160,203)'],
                    color='Type', 
                    #width=800, height=800, 
                    labels = {'exp_risk':'Expected Annual Risk', 
                            'exp_return': 'Expected Annual Return', 
                            'exp_esg': 'ESG score'})
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig

def plotting_projection_plotly(FA_3D_best, ef_R): 
    fig = px.scatter(FA_3D_best, x='exp_risk', y='exp_return', color='exp_esg',
                     color_continuous_scale='haline',
                    labels = {'exp_risk':'Expected Annual Risk', 
                              'exp_return': 'Expected Annual Return', 
                              'exp_esg': 'ESG risk score'})
    fig.add_trace(go.Scatter(x=ef_R[:,0], y=ef_R[:,1],
                        mode='lines',
                        name='Markowitz', 
                        marker_color='#222A2A'), 
                    )
    fig.update_layout(coloraxis_colorbar_y=0.45)
    return fig 

#########################################################################################################################################
#HISTOGRAMS  
def plot_histograms(FA_best, FA_3D_best): 
    fig, ax = plt.subplots(2,3, figsize= (12,4))
    ax[0][0].set_ylabel('2 objs portfolios')
    ax[1][0].set_ylabel('3 objs portfolios')
    ax[0][0].set_title('Expected Annual Risk')
    ax[0][1].set_title('Expected Annual Return')
    ax[0][2].set_title('ESG score')
    for i in range(3): 
        FA_best.iloc[:, i].hist(ax=ax[0][i])
        FA_3D_best.iloc[:, i].hist(ax=ax[1][i], color='darkgreen')
    return fig 

def plot_proportions(FA_best, FA_3D_best, X, best_assets, assets_info): 
    X_best = X[FA_best.index]
    X_3D_best = X[FA_3D_best.index]
    X_best_means = np.mean(X_best, axis=0)
    X_3D_best_means = np.mean(X_3D_best, axis=0)
    assets_proportion = assets_info.loc[best_assets.index]
    assets_proportion['Allocation 2obj']= X_best_means 
    assets_proportion['Allocation 3obj']= X_3D_best_means
    fig, ax = plt.subplots(ncols=2, figsize=(12,4))
    (assets_proportion[['Allocation 2obj','esg_score']]*100).plot(kind='barh', ax=ax[0])
    (assets_proportion[['Allocation 3obj','esg_score']]*100).plot(kind='barh', ax=ax[1])
    ax[0].set_title('Average Weights of Portfolios 2 objs')
    ax[1].set_title('Average Weights of Portfolios 3 objs')
    return fig 