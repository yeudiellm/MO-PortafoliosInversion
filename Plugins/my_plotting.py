import matplotlib.pyplot as plt
import seaborn as sns

def plot_assets(assets_info, best_assets): 
    assets_plot = assets_info.copy()
    assets_plot['Activos'] = 'Dominado'
    assets_plot['Size']    = 1000
    assets_plot['Profit'] = -assets_plot['Profit']
    assets_plot.loc[best_assets.index,'Activos']='No-Dominado'
    assets_plot.loc[best_assets.index,'Size'] = 3000
    fig = plt.figure(figsize=(10,8))
    g= sns.scatterplot(data=assets_plot, x='Risk', y='Profit', hue='Activos',
                    size='Size' ,palette=['cornflowerblue', 'red'])
    for idx, row in assets_plot.iterrows(): 
        plt.text(row['Risk'], row['Profit'], idx, fontsize=9)
    h,l = g.get_legend_handles_labels()
    plt.legend(h[0:3],l[0:3], loc='best', fontsize=13)
    plt.xlabel('Riesgo')
    plt.ylabel('Retorno Esperado')
    plt.show()