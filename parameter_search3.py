from datetime import datetime

import numpy as np
from scipy.stats import gmean

import ot
import wasserstein_smote

from sklearn.model_selection import KFold, train_test_split

from joblib import Parallel, delayed


def double_wasserstein1(X_train_smote):
    
    n, m, r, _ = X_train_smote.shape

    # uniform measures at points clouds of card m
    a2 = np.ones(m) / m
    b2 = np.ones(m) / m

    # uniform measures at points of card r
    a1 = np.ones(r) / r
    b1 = np.ones(r) / r

    # 1st level distance matrix of size m x m
    M1 = np.zeros((m, m))

    # M1 loop
    for i in range(m):
        for j in range(i+1, m):

            # pairwise squared Euclidean distances as the ground metric
            M0_ij = ot.dist(X_train_smote[0,i], X_train_smote[1,j], metric="sqeuclidean")

            # 2-Wasserstein distance btw point clouds, take square root
            M1[i,j] = ot.emd2(a1, b1, M0_ij) ** 0.5

    # 1st level symmetrize
    M1 = M1 + M1.T
    np.fill_diagonal(M1, 1e9)

    # 1-Wasserstein distance btw collections of point clouds
    W1 = ot.emd2(a2, b2, M1)
    
    return W1


def process_k(k, X_train, X_test, params, factor=100):
    
    # fit oversampler to X_test
    oversampler = wasserstein_smote.Oversampling(X_test, k=k, kernel=params["graph"], d=params["delta"], random_state=params["r"], n_jobs=4)

    # generate synthetic points to match X_train in size
    X_smote = oversampler.sample(method=params["method"], n=(X_train.shape[0],1))

    # merge train and SMOTE data points
    X_train_smote = np.concatenate((np.expand_dims(X_train, 0), np.expand_dims(X_smote, 0)), axis=0)

    # compute double Wasserstein distance
    return double_wasserstein1(X_train_smote) * factor


def compute(input_train, input_test, n, method, graph, k_min, k_max, k_num, delta, r, n_jobs, n_repeats):
    print("INPUT: {}\nNUMBER: {}\nMETHOD: {}\nGRAPH: {}\nk (min/max/num): {}, {}, {}\ndelta: {}\nrandom_state: {}\nn_jobs: {}".format(
        input_train[0], n,
        method, graph,
        k_min, k_max, k_num,
        delta,
        r,
        n_jobs))

    # params
    params = {
        "method": method,
        "graph": graph,
        "delta": delta,
        "r": r
    }

    # load train/test data
    X_tr = np.load(input_train[0]) # w300_train_3148.npy
    X_ts = np.load(input_test[0]) # w300_test_689.npy
    
    ks = []
    for i in np.linspace(k_min, k_max, k_num):
        ks.append(np.ceil(i).astype(int))

    print("take_n", n)
    print("KS", ks)

    print("# repeats: {}".format(n_repeats))

    factor = 100

    dw1 = np.zeros((n_repeats, k_num))

    for p in range(n_repeats):

        # randomly select n point clouds from test
        train_idx = np.random.choice(X_tr.shape[0], n, replace=False)

        # set train/test
        X_train = X_tr[train_idx]
        X_test = X_ts

        print("Repeat {}".format(p))

        # fill dw1 row
        dw1[p,:] = Parallel(n_jobs=n_jobs)(delayed(process_k)(j, X_train, X_test, params) for j in ks)
    
    print(np.mean(dw1, axis=0))

    # get date and time
    now = datetime.now()
    date_ = now.strftime("%d-%m-%Y")
    time_ = now.strftime("%H-%M-%S")
    
    # save to file
    filename = "./data/{}_{}_{}_{}_n_{}_kmin_{}_kmax_{}_knum_{}_delta_{}_r_{}_p_{}.npy".format(date_, time_, method, graph, n, k_min, k_max, k_num, delta, r, n_repeats)
    np.save(filename, dw1)

    #print("{:.3f}±{:.3f}".format(np.mean(dw1), np.std(dw1)))

if __name__ == "__main__":
    import argparse

    # set parser
    parser = argparse.ArgumentParser(
        description="Simplicial oversampling experiment, saves results to ./data directory",
        usage="python experiment.py -s ecoli"
    )

    # configure parser
    group1 = parser.add_argument_group("Input/output file names")
    group1.add_argument("-i_train", "--input_train", help='Train input, .npy file of landmarks', nargs='*', required=True)
    group1.add_argument("-i_test", "--input_test", help='Test input, .npy file of landmarks', nargs='*', required=True)
    group1.add_argument("-n", "--number", type=int, help="Number of landmarks to consider (if omitted consider all landmarks)", default=None)

    group2 = parser.add_argument_group("Algorithm configuration")
    group2.add_argument("-m", "--method", choices=["SMOTE", "simplicial", "simplicial_maximal"], help="Oversampling method, default='simplicial_maximal'", default="simplicial_maximal")
    group2.add_argument("-g", "--graph", choices=["knn", "cknn"], help="Oversampling method, default='cknn'", default="cknn")
    group2.add_argument("-k_min", "--k_min", type=int, help="Number k of neighbords to consider, min", default=True)
    group2.add_argument("-k_max", "--k_max", type=int, help="Number k of neighbords to consider, max", default=True)
    group2.add_argument("-k_num", "--k_num", type=int, help="Number k of neighbords to consider, num", default=True)
    group2.add_argument("-d", "--delta", type=float, help="Radius parameter delta, default=1.0", default=1.0)

    group3 = parser.add_argument_group("General parameters")
    group3.add_argument("-r", "--random_state", type=int, help="Random state", default=0)
    group3.add_argument("-p", "--n_repeats", type=int, help="Number of repeats", default=25)

    group4 = parser.add_argument_group("Number of parallel job processes")
    group4.add_argument("-j", "--n_jobs", type=int, help="Number of job processes, default=4", default=4)

    # parse command line
    args = parser.parse_args()

    # run
    compute(args.input_train, args.input_test, args.number, args.method, args.graph, args.k_min, args.k_max, args.k_num, args.delta, args.random_state, args.n_jobs, args.n_repeats)