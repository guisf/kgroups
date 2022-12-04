"""
Cluster some individual cases and create a graph plot.
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
    
import graph_clustering
import get_data

def community_layout(g, partition):
    #pos_communities = _position_communities(g, partition, scale=3.)
    pos_communities = _position_communities(g, partition, scale=10.)

    #pos_nodes = _position_nodes(g, partition, scale=1.)
    pos_nodes = _position_nodes(g, partition, scale=6.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    #pos_communities = nx.spring_layout(hypergraph, **kwargs)
    #os_communities = nx.spectral_layout(hypergraph, **kwargs)
    pos_communities = nx.circular_layout(hypergraph, **kwargs)
    #pos_communities = nx.shell_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        #pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos_subgraph = nx.circular_layout(subgraph, **kwargs)
        #pos_subgraph = nx.spectral_layout(subgraph, **kwargs)
        #pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def sbm_network(ax, n=1000, k=5, lamb=1.9, d=3,
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']):

    from metric import accuracy
    
    a = lamb*(k-1)*np.sqrt(d) + d 
    b = d - lamb*np.sqrt(d)
    probMatrix = a*np.eye(k) + (b*np.ones((k,k))-b*np.eye(k))
    A, z = get_data.SBM(n, k, probMatrix)
    
    # clustering
    zh = graph_clustering.bethe_hess(A, z, c=d)
    overlap_bethe = (k/(k-1))*(accuracy(z,zh) - 1./k)

    zh = graph_clustering.kgroups_bethe_hessian(A, z, c=d)
    overlap_kgroups = (k/(k-1))*(accuracy(z,zh) - 1./k)

    print("Bethe:", overlap_bethe, "k-Groups:", overlap_kgroups)
    
    partition = {i: k for i, k in enumerate(z)}
    #partition_hat = {i: k for i, k in enumerate(zh)}

    G = nx.from_numpy_matrix(A)
    pos = community_layout(G, partition)
    
    col_map = [colors[int(a)] for a in zh]
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col_map, 
                        node_size=65, width=0.1, linewidths=1, alpha=0.7)
    edges = nx.draw_networkx_edges(G, pos, ax=ax, width=0.1, edge_color='k',
                                    alpha=0.7)
    nodes.set_edgecolor('k')
    ax.axis('off')

    #ax.set_title(r'BH: %.2f, kG: %.2f'%(overlap_bethe, overlap_kgroups),fontsize=16)
    


###############################################################################
if __name__ == '__main__':

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    sbm_network(ax1, n=128, k=4, lamb=1.1, d=16)
    sbm_network(ax2, n=128, k=4, lamb=1.8, d=16)
    sbm_network(ax3, n=128, k=4, lamb=3.5, d=16)
    
    plt.subplots_adjust(wspace=0.1)

    fig.savefig('communities.pdf', bbox_inches='tight')

