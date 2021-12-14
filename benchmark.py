import time
import pickle
import numpy
import argparse
from random import sample
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Synthetic Sampling Benchmarks')

parser.add_argument('--g_file', type=str)
parser.add_argument('--B', type=int, default=300)
parser.add_argument('--L', type=int, default=3)
parser.add_argument('--k', type=int, default=5)
args = parser.parse_args()


def samp_neigh_g(u):
    if len(graph[u]) < K:
        return graph[u].copy()
    return set(sample(graph[u], K))

def seq_sample_nodes(vs, adj, n, l, k):
    node_sets = [set() for i in range(l)]
    node_sets[l - 1] = set(vs)
    for i in reversed(range(l - 1)):
        node_sets[i] = node_sets[i].union(node_sets[i + 1])
        samps = []
        for u in node_sets[i + 1]:
            neigh_sample = samp_neigh_g(u)
            samps.append(neigh_sample)
        node_sets[i] = node_sets[i].union(*samps)

    return node_sets


def parallel_sample_nodes(vs, adj, n, l, k):
    po = Pool(8)
    node_sets = [set() for i in range(l)]
    node_sets[l - 1] = set(vs)
    for i in reversed(range(l - 1)):
        node_sets[i] = node_sets[i].union(node_sets[i + 1])

        samps = po.map(samp_neigh_g, node_sets[i + 1])
        node_sets[i] = node_sets[i].union(*samps)

    po.close()
    return node_sets


if __name__ == '__main__':
    fname = args.g_file
    open_file = open(fname, "rb")
    graph = pickle.load(open_file)
    open_file.close()

    n = len(graph)
    L = args.L
    B = args.B
    k = args.k
    K = k
    # Sequential Standard
    print("Sequential Standard:")
    t1 = time.time()
    vs = sample(list(range(n)), B)
    ls = seq_sample_nodes(vs, graph, n, L, k)
    t2 = time.time()
    print(t2 - t1)
    for i in range(L):
        print("B(i) size:", len(ls[i]))
    print()

    # Parallel Standard
    print("Parallel Standard:")
    t1 = time.time()
    ls = parallel_sample_nodes(vs, graph, n, L, k)
    t2 = time.time()
    print(t2 - t1)
    for i in range(L):
        print("B(i) size:", len(ls[i]))

