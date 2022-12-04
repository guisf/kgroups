"""Construct table based on dermatology clustering.

We obtained the following:

+---------------------+----------------+----------------+
|      Algorithm      |    Accuracy    |     A-Rand     |
+---------------------+----------------+----------------+
|        kmeans       | 0.724043715847 | 0.69533698756  |
|         GMM         | 0.893442622951 |  0.8465538079  |
| spectral clustering | 0.953551912568 | 0.912517180432 |
|       spectral      | 0.950819672131 | 0.904961836633 |
|    kernel k-means   | 0.882513661202 | 0.861732888116 |
|   kernel k-groups   | 0.96174863388  | 0.935633109305 |
+---------------------+----------------+----------------+

kernel k-means and k-groups were initialized with k-means++


The number of points are:

[[112   0   0   0   0   0]
 [  0  61   0   0   0   0]
 [  0   0  72   0   0   0]
 [  0   0   0  49   0   0]
 [  0   0   0   0  52   0]
 [  0   0   0   0   0  20]]

The estimated number of points are:

[[ 51   0   0   0   0   0]
 [  0  52   0   0   0   0]
 [  0   0 112   0   0   0]
 [  0   0   0  59   0   0]
 [  0   0   0   0  72   0]
 [  0   0   0   0   0  20]]

These results are in the appropriate files inside ./data/

"""

# author: Guilherme Franca <guifranca@gmail.com>
# Johns Hopkinks University

import numpy as np
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns


import eclust

Z = pd.read_csv("data/dermatology_true_label_matrix.csv", header=None).values
Zh = pd.read_csv("data/dermatology_pred_label_matrix.csv", header=None).values

df = pd.read_csv('data/dermatology_data.csv', sep=',', header=None)
X = df.values[:,:-1]
missing_id = np.where(X=='?')
not_missing_id = [i for i in range(len(X)) if i not in missing_id[0]]
mean_age = np.array(X[not_missing_id,33], dtype=float).mean()
X[missing_id] = mean_age
X = np.array(X, dtype=float)

print("True number of points")
print(Z.T.dot(Z))

flatui = [
'#f1eef6',
'#d0d1e6',
'#a6bddb',
'#74a9cf',
'#2b8cbe',
'#045a8d',
]
cmap = sns.color_palette(flatui)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
im = sns.heatmap(Z.T.dot(Z), annot=True, vmin=0, vmax=112, fmt="d", 
            cmap=cmap, ax=ax1, cbar=False)
ax1.set_title('truth')
#fig.savefig('derma_heat_true.pdf')

print("Estimated number of points")
print(Zh.T.dot(Zh))

# estimate the maping based on the means
m = {}
rho = lambda x, y: np.power(np.linalg.norm(x-y), 0.5)
mu = []
for j in range(6):
    idx = np.where(Z[:,j]==1)[0]
    mu_j = X[idx,:].mean(axis=0)
    mu.append(mu_j)
mu = np.array(mu)
muh = []
for j in range(6):
    idx = np.where(Zh[:,j]==1)[0]
    muh_j = X[idx,:].mean(axis=0)
    muh.append(muh_j)
muh = np.array(muh)
for i in range(6):
    r = []
    for j in range(6):
        r.append(eclust.kernel_function(muh[i], mu[j], rho))
    j_star = np.argmax(r)
    m[i] = j_star

Zh2 = np.copy(Zh)
for i in range(6):
    Zh2[:,m[i]] = Zh[:,i]

print("Estimated number of points after finding the right map")
print(Zh2.T.dot(Zh2))

sns.heatmap(Zh2.T.dot(Zh2), annot=True, vmin=0, vmax=112, fmt="d", 
                cmap=cmap, ax=ax2, cbar=False)
ax2.set_title('estimated')

mappable = im.get_children()[0]
plt.colorbar(mappable, ax=[ax1, ax2], orientation='horizontal', aspect=40)

fig.savefig('derma_heat.pdf', bbox_inches='tight')


print()
print("Confusion matrix")
print()

t = PrettyTable(['class',1,2,3,4,5,6,'cases'])
true_clusters = []
estimated_clusters = []
for i in range(6):
    true_clusters.append(set(np.where(Z[:,i]==1)[0]))
    estimated_clusters.append(set(np.where(Zh2[:,i]==1)[0]))
i = 1
for c in true_clusters:
    row = [i]
    for ch in estimated_clusters:
        row.append(len(c.intersection(ch)))
    row.append(len(c))
    t.add_row(row)
    i += 1
row = ['total']
for c in estimated_clusters:
    row.append(len(c))
row.append(len(Zh))
t.add_row(row)
print()
print(t)

