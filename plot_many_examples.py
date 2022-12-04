"""Plot the results from many_examples.py"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

import sys

f = open('data/many_exp_kmeansplus.pickle', 'rb')
df = pickle.load(f)

output = 'data_many_kmeansplus.pdf'

#colors = sns.color_palette("deep", 3)
#colors = [
#    "#e41a1c",
#    "#377eb8",
#    "#4daf4a",
#]

colors = [
    "tab:red",
    "tab:blue",
    "tab:green",
]

#data_sets = np.unique(df['data'].values)
data_sets = [
    'libras_movement',
    'vehicle',
    'forest',
    'covertype',
    'segmentation',
    'wine', 
    'synthetic_control', 
    'fertility',
    'glass',
    'ionosphere',
    'epileptic',
    'anuran',
    'gene_expression',
    'pendigits',
    'seeds',
    'spect_heart',
    'iris', 
    'contraceptive',
#    'yeast',
#    'statlog'
]

fig, axs = plt.subplots(ncols=6, nrows=3, sharex=True, figsize=(25, 12))
axes = axs.flat

i = 0
#xranges = [[-0.3, 0.3], [0.7, 1.3], [1.7,2.3]]
xranges = [[-0.5, 2.5], [-0.5, 2.5], [-0.5,2.5]]
linestyles = ['-', '-', '-']
for data_set in data_sets:

    #if i in [0, 5]:
    #    i += 1
    
    ax = axes[i]

    dat = df.loc[df['data'] == data_set]
    means = [dat.loc[dat['algorithm']==alg]['nmi'].values.mean()
             for alg in ['k-groups', 'k-means', 'spectral']]
    meansstr = ['%.3f'%m for m in means]
    print(data_set, *meansstr)
    
    #for xs, ls, c, mean in zip(xranges, linestyles, colors, means):
    #    ys = [mean]*len(xs)
    #    ax.plot(xs, ys, linewidth=4, color=c, linestyle=ls, alpha=0.3)
    
    sns.violinplot(x='algorithm', y='nmi', data=dat, scale='count', 
                   palette=colors, ax=ax, saturation=.75)
    #plt.setp(ax.collections, alpha=.7)
    
    #ax = sns.swarmplot(x='algorithm', y='nmi', data=dat, 
    #            palette=colors, ax=ax)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(data_set.replace('_', ' '))
    
    i += 1

axes[0].set_ylabel('NMI')
axes[6].set_ylabel('NMI')
axes[12].set_ylabel('NMI')
axes[12].set_xticklabels(['groups', 'means', 'spectral'])

#fig.delaxes(axes[0])
#fig.delaxes(axes[5])

#lines = [Line2D([0], [0], color=c, linewidth=5, linestyle='-') 
#            for c in colors[:3]]
#labels = ['kernel k-groups', 'kernel k-means', 'spectral clustering']
#plt.figlegend(lines, labels, loc=(0.806,0.1))

plt.subplots_adjust(wspace=0.22, hspace=0.14)

fig.savefig(output, bbox_inches='tight')


