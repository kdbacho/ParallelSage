import time
import pickle
import numpy
import argparse

parser = argparse.ArgumentParser(description='Generate Simple Synthetic Graphs')

parser.add_argument('--name', type=str, default='graph')
parser.add_argument('--n', type=int, default=10000)
parser.add_argument('--p', type=float, default=0.1)
args = parser.parse_args()


'''
# Generate sample graphs
n : Number of vertices
p : probability that two distinct vertices are connected

Returns adj: Adjacency list representation of the graph
'''
def gen_graph(n, p):
    adj = [set() for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if numpy.random.binomial(1,p):
                adj[i].add(j)
                adj[j].add(i)
        if (i + 1) % 100 == 0:
            print(i + 1, "completed")
                
    return adj


if __name__ == '__main__':
    fname = args.name
    p = args.p
    n = args.n

    graph = gen_graph(n, p)
    print("Graph Created")
    open_file = open(fname + ".pkl", "wb")
    pickle.dump(graph, open_file)
    open_file.close()
    print("Graph Created, saved as", fname + ".pkl")

