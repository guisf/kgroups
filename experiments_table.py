import pandas as pd
import numpy as np
from prettytable import PrettyTable
import sys

#df = pd.read_csv('results1.csv', sep=',', header=None)
#df = pd.read_csv('results3.csv', sep=',', header=None)
#df = pd.read_csv('results4.csv', sep=',', header=None)
#df = pd.read_csv('results5.csv', sep=',', header=None)
#df = pd.read_csv('results6.csv', sep=',', header=None)
df = pd.read_csv('results7.csv', sep=',', header=None)

# results for the first metric only
score1 = [0,0,0]
score2 = [0,0,0]
names = []
for index, row in df.iterrows():
    name = row[0]
    names.append(name)
    sizes = row[[1,2,3]].values
    metrics = row[[4, 6, 8, 10, 12, 14]].values
    accuracy = row[[4,8,12]].values
    nmi = row[[6,10,14]].values
    j1 = accuracy.argmax()
    j2 = nmi.argmax()
    score1[j1] += 1
    score2[j2] += 1
    print(' & '.join([name]+['%i'%x for x in sizes]+['%.3f'%x for x in metrics]))
print(score1)
print(score2)
print()

for index, row in df.iterrows():
    name = row[0]
    names.append(name)
    sizes = row[[1,2,3]].values
    stds = row[[5, 7, 9, 11, 13, 15]].values
    print(' & '.join([name]+['%.3E'%x for x in stds]))


sys.exit()

df = pd.read_csv('results2.csv', sep=',', header=None)
score1 = [0,0,0]
score2 = [0,0,0]
for name in names:
    
    res = []
    sizes = df[df[0]==name][[1,2,3]].values[0]
    a = df[df[0]==name][[4,6,8,10,12,14]].values
    for i in range(6):
        j = a[:,i].argmax()
        res.append(a[j,i])
    res = np.array(res)
    k = res[[0,2,4]].argmax()
    score1[k] += 1
    k = res[[1,3,5]].argmax()
    score2[k] += 1

    print(' & '.join([name]+['%i'%x for x in sizes]+['%.3f'%x for x in res]))
print(score1)
print(score2)



sys.exit()


for f in files:
    df = pd.read_csv(f, sep=',', header=None)
    dfs.append(df)

score = [0, 0, 0]
t = PrettyTable(['name', 'k-groups', 'k-means', 'spectral', 'win'])
kgroups_win = []
for index, row in dfs[0].iterrows():
        name = row[0]
        acc = [row[4], row[8], row[12]]
        j = np.argmax(acc)
        score[j] += 1
        t.add_row([name] + ['%.3f'%x for x in acc] + [j])
        if j == 0:
            kgroups_win.append(name)
print(t)
print(score)
print(kgroups_win)

names = dfs[0][0].values
df2 = pd.concat(dfs)
score = [0, 0, 0]
kgroups_win = []
t = PrettyTable(['name', 'k-groups', 'k-means', 'spectral', 'win', 'metric'])
for name in names:
    a = df2[df2[0]==name]
    a = a[[4,8,12]].values
    best_metric = [a[:,i].argmax() for i in range(3)]
    best_value = [a[:,0].max(), a[:,1].max(), a[:,2].max()]
    j = np.argmax(best_value)
    score[j] += 1
    t.add_row([name] + ['%.3f'%x for x in best_value] + [j, best_metric[j]])
    if j == 0:
        kgroups_win.append(name)
print(t)
print(score)
print(kgroups_win)

