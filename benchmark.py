import time
import pickle
import numpy
import argparse
from math import floor, log, ceil
from random import sample
from multiprocessing import Pool
from tree import Tree

parser = argparse.ArgumentParser(description='Synthetic Sampling Benchmarks')

parser.add_argument('--g_file', type=str)
parser.add_argument('--B', type=int, default=300)
parser.add_argument('--L', type=int, default=3)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--procs', type=int, default=8)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--sb_acc_trials', type=int, default=0)
args = parser.parse_args()


def samp_neigh_g(u):
    if len(graph[u]) < K:
        return graph[u].copy()
    return set(sample(graph[u], K))


def seq_sample_nodes(vs, l):
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


def parallel_sample_nodes(vs, l):
    po = Pool(args.procs)
    node_sets = [set() for i in range(l)]
    node_sets[l - 1] = set(vs)
    for i in reversed(range(l - 1)):
        node_sets[i] = node_sets[i].union(node_sets[i + 1])

        samps = po.map(samp_neigh_g, node_sets[i + 1])
        node_sets[i] = node_sets[i].union(*samps)

    po.close()
    return node_sets


def tree_size(h):
    return (K ** (h + 1) - K) // (K - 1)

def tree_sample(b, B, l, parallel=True):
    leaf_set = set()
    batch_set = set()
        
    def sample_tree(v, curr_l):
        batch_set.add(v)
        if curr_l > 0:
            for u in samp_neigh_g(v):
                sample_tree(u, curr_l - 1)
        elif curr_l == 0:
            leaf_set.add(v)
  
    f = floor((log(1 + B * (K - 1)) / log(K)) - 1)
    
    sample_tree(b, f)

    if len(batch_set) < B:
        # Adjust Leaves
        remain = B - len(batch_set)
        num_to_expand = ceil(remain / K)
        if num_to_expand < len(leaf_set):
            to_expand = sample(leaf_set, num_to_expand)
        else:
            to_expand = list(leaf_set)
        expans = []
        for u in to_expand:
            leaf_set.remove(u)
            expans.append(set(samp_neigh_g(u)))
        batch_set = batch_set.union(*expans)
        leaf_set = leaf_set.union(*expans)
    non_leafs = batch_set - leaf_set
    if parallel:
        node_sets = parallel_sample_nodes(leaf_set, l)
    else:
        node_sets = seq_sample_nodes(leaf_set, l)

    for i in range(l):
        node_sets[i] = node_sets[i].union(non_leafs)
    
    return  node_sets, batch_set, leaf_set




if __name__ == '__main__':
    fname = args.g_file
    open_file = open(fname, "rb")
    graph = pickle.load(open_file)
    open_file.close()

    n = len(graph)
    L = args.L
    B = args.B
    K = args.k
    T = args.trials

    seq_s = 0
    par_s = 0
    seq_tree = 0
    par_tree = 0

    # Sequential Standard
    print("Sequential Standard:")
    tot_time = 0
    for i in range(T):
        t1 = time.time()
        vs = sample(list(range(n)), B)
        ls = seq_sample_nodes(vs, L)
        t2 = time.time()
        tot_time += (t2 - t1)
    seq_s = tot_time / T
    print(tot_time, tot_time / T)
    for i in range(L):
        print("B(" + str(i) + ") size:", len(ls[i]))
    print()

    # Parallel Standard
    print("Parallel Standard:")
    tot_time = 0 
    for i in range(T):
        t1 = time.time()
        ls = parallel_sample_nodes(vs, L)
        t2 = time.time()
        tot_time += (t2 - t1)
    par_s = tot_time / T
    print(tot_time, tot_time /T)
    for i in range(L):
        print("B(" + str(i) + ")size:", len(ls[i]))
    print()

    # Tree
    print("Tree Sequential:")
    tot_time = 0
    for i in range(T):
        t1 = time.time()
        v = sample(list(range(n)),1)[0]
        ls, bs, leafs = tree_sample(v, B, L, parallel=False)
        t2 = time.time()
        tot_time += (t2 - t1)
    seq_tree = tot_time / T
    print(tot_time, tot_time / T)
    for i in range(L):
        print("B(" + str(i) + ") size:", len(ls[i]))
    print("effective_batch_size:", len(bs))
    print("computation_batch_size:", len(leafs))

    # Tree
    print("Tree Parallel:")
    tot_time = 0
    for i in range(T):
        t1 = time.time()
        v = sample(list(range(n)),1)[0]
        ls, bs, leafs = tree_sample(v, B, L, parallel=True)
        t2 = time.time()
        tot_time += (t2 - t1)
    par_tree = tot_time / T
    print(tot_time, tot_time / T)
    for i in range(L):
        print("B(" + str(i) + ") size:", len(ls[i]))
    print("effective_batch_size:", len(bs))
    print("computation_batch_size:", len(leafs))

    print("Speedups", (seq_s / par_s), (seq_s / seq_tree), (seq_s / par_tree))

    if args.sb_acc_trials != 0:
        print("Testing Sb accuracy and shrinkage")
        num_t = args.sb_acc_trials 
        tot_rel_error = 0
        tot_shrink = 0
        for i in range(args.sb_acc_trials):
            v= sample(list(range(n)), 1)[0]
            # Depth of network doesn't matter here
            ls, bs, leafs = tree_sample(v,B,1,parallel=True)
            tot_rel_error += abs(B - len(bs)) /len(bs)
            tot_shrink += B / len(leafs)
        print("avg rel error", tot_rel_error / num_t)
        print("avg shrink", tot_shrink / num_t)



