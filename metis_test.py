import networkx as nx #install v2.3.0, pip3 install networkx=2.3.0
import metis
from networkx.drawing.nx_pydot import write_dot

#G = nx.Graph()
#G.add_edges_from([(3,1),(2,3),(1,2),(3,4),(4,5),(5,6),(5,7),(7,6),(4,10),(10,8),(10,9),(8,9)])
G = metis.example_networkx()
(edgecuts, parts) = metis.part_graph(G, 3)
colors = ['red', 'blue', 'green']
for i, p in enumerate(parts):
    G.node[i]['color'] = colors[p]

write_dot(G, 'example.dot') #use dot -Tps example.dot -o outfile.ps
