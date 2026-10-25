# Install required libraries (run in terminal / command prompt):
# pip install networkx matplotlib

import networkx as nx
import matplotlib.pyplot as plt

edges = [
    ("J","G"),("G","K"),("J","F"),("F","G"),("G","H"),("H","K"),
    ("F","C"),("F","A"),("H","B"),("H","I"),("A","B"),
    ("C","A"),("A","E"),("C","D"),("D","E"),("C","E"),("A","D"),
    ("B","I"),("I","L"),("L","M"),("M","B"),("B","L"),("I","M")
]

g = nx.Graph(edges)
pos = {
    "J":(1,3), "G":(2,3), "K":(3,3), "F":(1,2), "H":(3,2),
    "C":(0,1), "A":(1,1), "B":(2,1), "I":(3,1),
    "D":(0,0), "E":(1,0), "L":(2,0), "M":(3,0)
}

print("Nodes:", g.number_of_nodes(), "\nEdges:", g.number_of_edges())
plt.figure()
nx.draw(g, pos, with_labels=True, node_color="lightblue")
plt.title("Original Network")
plt.show()

original = g.copy()
mods, results = [], []
communities = [set(c) for c in nx.connected_components(g)]

while g.number_of_edges():
    edge = max(nx.edge_betweenness_centrality(g), key=nx.edge_betweenness_centrality(g).get)
    g.remove_edge(*edge)
    communities = [set(c) for c in nx.connected_components(g)]
    m = nx.community.modularity(original, communities)
    mods.append(m)
    results.append(communities)
    print("Removed:", edge, "| Communities:", len(communities),
          "| Modularity:", round(m, 3))

best = mods.index(max(mods))
communities = results[best]

print("\nBest iteration:", best + 1)
print("Maximum modularity:", round(mods[best], 3))
for i, c in enumerate(communities, 1):
    print("Community", i, ":", sorted(c))

colors = {}
for i, c in enumerate(communities):
    for node in c:
        colors[node] = i

plt.figure()
nx.draw(original, pos, with_labels=True,
        node_color=[colors[n] for n in original.nodes()])
plt.title("Detected Communities")
plt.show()

plt.plot(range(1, len(mods) + 1), mods, marker="o")
plt.scatter(best + 1, mods[best], color="red")
plt.xlabel("Iteration")
plt.ylabel("Modularity")
plt.title("Girvan-Newman Modularity")
plt.show()