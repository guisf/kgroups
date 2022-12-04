
import numpy as np
import pandas as pd
import networkx as nx
import urllib.request as urllib
import io
import zipfile

def yeast():
    # normalization doesn't change much
    df = pd.read_csv('data/yeast.data', sep=',', header=None)
    classes = {'CYT': 0, 'NUC': 1, 'MIT': 2, 'ME3': 3, 'ME2': 4, 'ME1': 5,
           'EXC': 6, 'VAC': 7, 'POX': 8, 'ERL': 9}
    data = df.values[:,1:-1]
    z = df.values[:,-1]  # 3 classes, starting with 0
    z = np.array([classes[name] for name in z])
    z = np.array(z, dtype=int)
    data = np.array(data, dtype=float)
    idx = np.random.choice(range(len(data)), 2000)
    data = data[idx]
    z = z[idx]
    #data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data, z

def wine():
    # works best normalized
    df = pd.read_csv('data/wine.data', sep=',', header=None)
    data = df.values[:,1:]
    z = df.values[:,0]  # 3 classes, starting with 1
    z = z - 1
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    z = np.array(z, dtype=int)
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data, z

def wave():
    df = pd.read_csv('data/waveform-+noise.data', sep=',', header=None)
    data = df.values[:,:-1]
    z = df.values[:,-1]  # 3 classes, starting with 0
    #idx = np.random.choice(range(len(data)), 2000)
    #data = data[idx]
    #z = z[idx]
    z = np.array(z, dtype=int)
    data = (data - data.mean(axis=0))/data.std(axis=0)
    return data, z

def epileptic():
    # works best normalized
    df = pd.read_csv('data/epileptic.csv', sep=',', header=None)
    data = df.values[:,1:-1]
    data = np.array(data, dtype=float)
    z = df.values[:,-1]  # 3 classes, starting with 0
    z = np.array(z, dtype=int)
    z = z-1
    idx = np.random.choice(np.arange(0, len(z)), 2000)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data, z

def seeds():
    df = pd.read_csv('data/seeds_dataset.txt', sep=',', header=None)
    data = df.values[:,:-1]
    z = df.values[:,-1]  # 3 classes, starting with 1
    z = z - 1
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    z = np.array(z, dtype=int)
    data = (data - data.mean(axis=0))/data.std(axis=0)
    return data, z

def libras_movement():
    # doesn't change much normalized or not. normalized seems to improve 
    # spectral clustering only
    df = pd.read_csv('data/movement_libras.data', sep=',', header=None)
    data = df.values[:,:-1]
    z = df.values[:,-1]  # 3 classes, starting with 0
    z = np.array(z, dtype=int)
    z = z-1
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data, z

def synthetic_control():
    # works best normalized
    f = open('data/synthetic_control.data', 'r')
    data = []
    for line in f:
        numbers = [float(n) for n in line.split()]
        data.append(numbers)
    data = np.array(data)
    z = np.zeros(data.shape[0], dtype=int)
    z[0:100] = 0
    z[100:200] = 1
    z[200:300] = 2
    z[300:400] = 3
    z[400:500] = 4
    z[500:600] = 5
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data[idx], z[idx]

def anuran():
    # seems better normalized
    df = pd.read_csv('data/Frogs_MFCCs.csv', sep=',', header=0)
    data = df.values[:,:-4]
    data = np.array(data, dtype=float)
    species = df.values[:,-2]
    specie_names = np.unique(species)
    d = {}
    for i, n in enumerate(specie_names):
        d[n] = i
    z = np.array([d[name] for name in species], dtype=int)
    idx = np.random.choice(range(len(data)), 2000)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data, z

def iris():
    # better not normalized
    df = pd.read_csv('data/iris.data', sep=',', header=None)
    data = df.values[:,:-1]
    data = np.array(data, dtype=float)
    species = df.values[:,-1]
    specie_names = np.unique(species)
    d = {}
    for i, n in enumerate(specie_names):
        d[n] = i
    z = np.array([d[name] for name in species], dtype=int)
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    #data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data, z

def contraceptive():
    # better normalized, the results are not that great though
    df = pd.read_csv('data/cmc.data', sep=',', header=None)
    data = df.values[:,:-1]
    data = np.array(data, dtype=float)
    z = df.values[:,-1]
    z = z-1
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data, z

def covertype():
    # k-groups does better than others without normalization
    # but the results are worse overall
    df = pd.read_csv('data/covtype.data', sep=',', header=None)
    data = df.values[:,:-1]
    data = np.array(data, dtype=float)
    z = df.values[:,-1]
    z = z-1

    new_data = []
    new_z = []
    num_items = 200
    classes = np.unique(z)
    for k in classes:
        idx = np.where(z==k)[0]
        idx = np.random.choice(idx, num_items)
        for i in idx:
            new_data.append(data[i])
            new_z.append(k)
    new_data = np.array(new_data, dtype=float)
    new_z = np.array(new_z, dtype=int)
    
    idx = np.arange(new_data.shape[0])
    np.random.shuffle(idx)
    data = new_data[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    z = new_z[idx]
    return data, z

def fertility():
    # better not normalized, the results are not great 
    df = pd.read_csv('data/fertility_Diagnosis.txt', sep=',', header=None)
    data = df.values[:,:-1]
    data = np.array(data, dtype=float)
    labels = df.values[:,-1]
    d = {'N':1, 'O':0}
    z = np.array([d[name] for name in labels], dtype=int)
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    #data = (data - data.mean(axis=0))/(data.std(axis=0)+0.000001)
    return data, z

def forest():
    # works better when normalized
    df = pd.read_csv('data/forest.csv', sep=',', header=0)
    data = df.values[:,1:]
    data = np.array(data, dtype=float)
    classes = df.values[:,0]
    class_names = np.unique(classes)
    d = {}
    for i, n in enumerate(class_names):
        d[n] = i
    z = np.array([d[name] for name in classes], dtype=int)
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.00001)
    return data, z

def gene_expression():
    # better normalized
    df = pd.read_csv('data/TCGA_data.csv', sep=',', header=0)
    data = df.values[:,1:201]
    data = np.array(data, dtype=float)
    df = pd.read_csv('data/TCGA_labels.csv', sep=',', header=0)
    labels = df.values[:,1]
    class_names = np.unique(labels)
    d = {}
    for i, n in enumerate(class_names):
        d[n] = i
    z = np.array([d[name] for name in labels], dtype=int)
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.00001)
    return data, z

def glass():
    # works better without normalization
    df = pd.read_csv('data/glass.data', sep=',', header=None)
    data = df.values[:,1:-1]
    z = df.values[:,-1]
    labels = np.unique(z)
    d = {}
    for i, n in enumerate(labels):
        d[n] = i
    z = np.array([d[n] for n in z], dtype=int)
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    #data = (data - data.mean(axis=0))/(data.std(axis=0)+0.00001)
    return data, z

def ionosphere():
    # improves a little when normalized, but not much
    # kernel k-groups works best than other unnormalized
    df = pd.read_csv('data/ionosphere.data', sep=',', header=None)
    data = df.values[:,:-1]
    z = df.values[:,-1]
    data = np.array(data, dtype=float)
    labels = np.unique(z)
    d = {}
    for i, n in enumerate(labels):
        d[n] = i
    z = np.array([d[n] for n in z], dtype=int)
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    #data = (data - data.mean(axis=0))/(data.std(axis=0)+0.0000001)
    return data, z

def poker():
    df = pd.read_csv('data/poker-hand-training-true.data', sep=',', header=None)
    #df = pd.read_csv('data/poker_full.data', sep=',', header=None)
    data = df.values[:,:-1]
    data = np.array(data, dtype=float)
    z = np.array(df.values[:,-1], dtype=int)
    labels = np.unique(z)
    d = {}
    for i, n in enumerate(labels):
        d[n] = i
    z = np.array([d[n] for n in z], dtype=int)
    idx = np.random.choice(np.arange(data.shape[0]), 2000)
    data = data[idx]
    z = z[idx]
    return data, z

def spect_heart():
    # works better for spectral clustering when normalized
    # for k-groups works best unnormalized
    df = pd.read_csv('data/SPECT.data', sep=',', header=None)
    data = df.values[:,1:]
    data = np.array(data, dtype=float)
    z = np.array(df.values[:,0], dtype=int)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    #data = (data - data.mean(axis=0))/(data.std(axis=0)+0.0000001)
    return data, z

def pendigits():
    # the data seems pretty balanced.  There are ~ 7000 data points
    # 10 clusters
    df = pd.read_csv('data/pendigits.tra', sep=',', header=None)
    data = df.values[:,:-1]
    data = np.array(data, dtype=float)
    z = np.array(df.values[:,-1], dtype=int)
    idx = np.random.choice(range(len(data)), 2000)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.0000001)
    return data, z

def statlog():
    # 6 clusters
    df = pd.read_csv('data/sat.trn', sep=',', header=None)
    data = df.values[:,:-1]
    data = np.array(data, dtype=float)
    z = np.array(df.values[:,-1], dtype=int)
    z = z-1
    z[z==6] = 5
    idx = np.random.choice(range(len(data)), 2000)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.0000001)
    return data, z

def vehicle():
    # 4 clusters
    df = pd.read_csv('data/vehicle.dat', sep=',', header=None)
    data = df.values[:,:-1]
    data = np.array(data, dtype=float)
    z = np.array(df.values[:,-1], dtype=int)
    #data = (data - data.mean(axis=0))/(data.std(axis=0)+0.0000001)
    return data, z

def segmentation():
    # better normalized
    # 7 clusters
    df = pd.read_csv('data/segmentation.data', sep=',', header=None)
    data = df.values[:,1:]
    data = np.array(data, dtype=float)
    z = df.values[:,0]
    names = np.unique(z)
    for i, name in enumerate(names):
        z[z==name] = i
    z = np.array(z, dtype=int)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.0000001)
    return data, z

def wholesale():
    # first two cols are channel and region
    # if cluster on channel -> k=2
    # if cluster on region -> k=3
    df = pd.read_csv('data/wholesale.csv', sep=',', header=None)
    data = df.values[:,2:]
    data = np.array(data, dtype=float)
    z = np.array(df.values[:,0], dtype=int)
    z = z-1
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    data = (data - data.mean(axis=0))/(data.std(axis=0)+0.0000001)
    return data, z


###############################################################################
# graph clustering
    
def weird_adj_to_numpy(A): 
    """Correct weird numpy array given by networkx"""
    B = np.zeros(A.shape)
    idx = np.where(A!=0)
    B[idx] = 1
    return B

def eucore():
    # creating adjacency matrix
    df = pd.read_csv('data/email-Eu-core.txt', sep=' ', header=None)
    data = df.values[:,:]
    vertex = np.unique(data[:,0])
    vertex2 = np.unique(data[:,1])
    n  = np.max([vertex.max(), vertex2.max()])
    A = np.zeros(shape=(n+1,n+1))
    for e in data:
        i, j = e
        A[i,j] = 1.
    # creating labels
    df = pd.read_csv('data/email-Eu-core-department-labels.txt',
                    sep=' ',header=None)
    data = df.values[:,:]
    z = np.zeros(A.shape[0], dtype=int)
    for e in data:
        i, label = e
        z[i] = label
    return A, z
    
def karate():
    #G = nx.read_gml('data/karate.gml', label='id')
    #z = [prop['class'] for name, prop in G.nodes(data=True)]
    G = nx.read_gml('data/real_world_networks/karate.gml', label='id')
    z = [prop['value'] for name, prop in G.nodes(data=True)]
    A = nx.to_numpy_matrix(G)
    return weird_adj_to_numpy(A), z
    
def dolphin():
    #G = nx.read_gml('data/dolphins.gml')
    #z = [sex['class'] for name, sex in G.nodes(data=True)]
    G = nx.read_gml('data/real_world_networks/dolphins.gml')
    z = [prop['value'] for name, prop in G.nodes(data=True)]
    A = nx.to_numpy_matrix(G)
    return weird_adj_to_numpy(A), z

def polbooks():
    #G = nx.read_gml('data/polbooks.gml')
    #z = [category['value'] for book, category in G.nodes(data=True)]
    #translate = {name: i for i, name in enumerate(np.unique(z))}
    #z = [translate[label] for label in z]
    G = nx.read_gml('data/real_world_networks/polbooks.gml')
    z = [prop['value']-1 for name, prop in G.nodes(data=True)]
    A = nx.to_numpy_matrix(G)
    return weird_adj_to_numpy(A), z

def football():
    G = nx.read_gml('data/football.gml')
    z = [conference['value'] for team, conference in G.nodes(data=True)]
    # this one has edges messed up compared to the nodes, i.e.
    # the enumeration do not match
    #G = nx.read_gml('data/real_world_networks/football.gml', label=id)
    #z = [prop['value'] for name, prop in G.nodes(data=True)]
    A = nx.to_numpy_matrix(G)
    return weird_adj_to_numpy(A), z

def condmat():
    G = nx.read_gml('data/cond-mat.gml', label='id')
    A = nx.to_numpy_matrix(G)
    return weird_adj_to_numpy(A)

def polblogs():
    G = nx.read_gml('data/polblogs.gml', label='id')
    z = [category['value'] for book, category in G.nodes(data=True)]
    translate = {name: i for i, name in enumerate(np.unique(z))}
    z = [translate[label] for label in z]
    G = G.to_undirected()
    A = nx.to_numpy_matrix(G)
    return weird_adj_to_numpy(A), z

def polblogs2():
    import pickle
    f = open('data/real_world_networks/polblogs.gml', 'r')
    nodes = []
    labels = []
    sources = []
    targets = []
    for line in f:
        line = line.strip()
        if line.startswith('id'):
            nodes.append(int(line.split()[1]))
        elif line.startswith('value'):
            labels.append(int(line.split()[1]))
        elif line.startswith('source'):
            sources.append(int(line.split()[1]))
        elif line.startswith('target'):
            targets.append(int(line.split()[1]))
    sources = np.array(sources)-1
    targets = np.array(targets)-1
    A = np.zeros((len(nodes), len(nodes)))
    for i in sources:
        for j in targets:
            A[i,j] = 1
    labels = np.array(labels, dtype=int)
    pickle.dump([A, labels], open('data/polblogs_Az.pickle', 'wb'))

def SBM(n, k, Q):
    """Generate adjacency matrix and labels for a stochastic block model."""
    labels = np.random.randint(0, k, size=n)
    P = np.array(Q)/n
    A = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            p = P[labels[i], labels[j]]
            epsilon = np.random.rand()
            if p >= epsilon:
                A[i,j] = A[j,i] = 1
    return A, labels

def SBM2(n, k, Q):
    """Generate adjacency matrix and labels for a stochastic block model.
    This one has a different scale of the probability matrix and
    makes recovery much easier.
    
    """
    labels = np.random.randint(0, k, size=n)
    P = np.array(Q)*np.log(n)/n
    A = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            p = P[labels[i], labels[j]]
            epsilon = np.random.rand()
            if p >= epsilon:
                A[i,j] = A[j,i] = 1
    return A, labels

def drosophila_left():
    df1 = pd.read_csv('data/left_adjacency.csv', sep=' ', header=None)
    #df2 = pd.read_csv('data/left_cell_labels.csv', sep=' ', header=None)
    A = df1.values
    return A

def drosophila_right():
    df1 = pd.read_csv('data/right_adjacency.csv', sep=' ', header=None)
    #df2 = pd.read_csv('data/left_cell_labels.csv', sep=' ', header=None)
    A = df1.values
    return A

def GrQc():
    """
    Arxiv GR-QC (General Relativity and Quantum Cosmology) collaboration network is from the e-print arXiv and covers scientific collaborations between authors papers submitted to General Relativity and Quantum Cosmology category. If an author i co-authored a paper with author j, the graph contains a undirected edge from i to j. If the paper is co-authored by k authors this generates a completely connected (sub)graph on k nodes.

The data covers papers in the period from January 1993 to April 2003 (124 months). It begins within a few months of the inception of the arXiv, and thus represents essentially the complete history of its GR-QC section.

    . Leskovec, J. Kleinberg and C. Faloutsos. Graph Evolution: Densification and Shrinking Diameters. ACM Transactions on Knowledge Discovery from Data (ACM TKDD), 1(1), 2007.
    """
    df = pd.read_csv('data/ca-GrQc.txt', sep='\t', header=None)
    data = df.values[:,:]
    x = np.sort(np.unique(data[:,0]))
    n = len(x)
    nodes = {node: i for i, node in enumerate(x)} # dictionary
    A = np.zeros((n,n), dtype=int)
    for a, b in data:
        A[nodes[a], nodes[b]] = 1
        A[nodes[b], nodes[a]] = 1
    return A

    


###############################################################################
if __name__ == '__main__':

    import sys
    import eclust
    import wrapper
    from sklearn.metrics import normalized_mutual_info_score as nmi
    
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    import sys

    #GrQc()
    polblogs2()

    sys.exit()

    sizes = [5, 5, 10]
    probs = [[0.25, 0.05, 0.02],
             [0.05, 0.35, 0.07],
             [0.02, 0.07, 0.40]]
    A, z = stochastic_block_model(sizes, probs)

    #X, z = seeds()
    #X, z = epileptic()
    #X, z = spect_heart()
    #X, z = segmentation()
    #X, z = vehicle()
    X, z = wholesale()

    k = len(np.unique(z))
    print(X.shape, k)

    rho = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*2))
    G = eclust.kernel_matrix(X, rho)

    res = []
    nt = 5
    zh = wrapper.kernel_kgroups(k, X, G, run_times=nt, ini='k-mean++')[0]
    print('k-groups', nmi(z, zh))
    zh = wrapper.kernel_kmeans(k, X, G, run_times=nt, ini='k-mean++')[0]
    print('k-means', nmi(z, zh))
    zh = wrapper.spectral_clustering(k, X, G, run_times=nt)[0]
    print('spectral', nmi(z, zh))
