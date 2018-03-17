import sys
sys.path.append('../src')

import exposureModelNets as expModel
import numpy as np
import networkx as nx


connected = False

print ("Generating the graph...")
## generate a connected graph
graph_type = 'barabasi_albert'
while not (connected):
    
    G = expModel.inferenceGraph()
    G .generate (1000, 3., graph_type = graph_type )
    connected = nx.is_connected ( G )


## get the number of nodes
## and the average degree
N = len(G.nodes())
k = np.mean(G.degree().values())

## fix a probability
## and an observation fraction
## to generate the perturbation
p = round(1./float(k), 2)+0.04
ob=0.1


print ("Generating the perturbation...")
## generate the perturbation
## in the graph
G.create_perturbation(p, ob)

## do belief propagation
print ("Performing belief propagation...")
G.full_bp ( max_iter = 20 )
## do label propagation
print ("Performing label propagation...")
G.label_propagation ( max_iter = 20 )
## do shortest paths
print ("Performing shortest paths inference...")
G.shortest_paths ()

## compute the AUC
print ("Computing AUC...")
auc_bp, auc_lp, auc_sp = G.auc()

print ("For a %s graph with %d nodes, average degree %g and %g%% observed nodes," %(graph_type, N, k, ob*100))
for auc, name in zip([auc_bp, auc_lp, auc_sp],\
                     ['Belief propagation', 'Label propagation', 'Shortest paths']):

    print ("\tachieved AUC %g with method %s" %(auc, name))