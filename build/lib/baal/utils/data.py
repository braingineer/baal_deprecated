import os
import pickle
import numpy as np


def get_pickle(fn, t='rb'):
    with open(fn, t) as fp:
        ret = pickle.load(fp)
    return ret


def save_pickle(outdir, save_name, obj, usetp=False):
    if usetp:
        magis.TOP_PATH + "/" + outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outdir + "/" + save_name + ".pkl", 'wb') as fp:
        pickle.dump(obj, fp)


def split_list(data, proportions):
    """
    for i,p in enumerate(proportions):
        lower_bound = N*sum(proportions[:i])
        upper_bound = lower_bound+p*N
        yield lower_bound,upper_bound
    """
    N = len(data)
    uptonow = lambda i: sum(proportions[:i])
    lower_bound = lambda i: int(N * uptonow(i))
    upper_bound = lambda i, p: int(N * (uptonow(i) + p))
    splitter = [[lower_bound(i), upper_bound(i, p)]
                for i, p in enumerate(proportions)]
    return [data[i:j] for i, j in splitter]


def make_kth_slice(data, k, k_size):
    """
    Input: data matrix, the current slice index and the size of each slice,
    Outpt: Data minus kth slice, kth slice of data
    """
    indices = np.ones(len(data), dtype=bool)
    indices[k * k_size:(k + 1) * k_size] = np.zeros(k_size, dtype=bool)
    kth_slice = np.zeros(len(data), dtype=bool)
    kth_slice[k * k_size:(k + 1) * k_size] = np.ones(k_size, dtype=bool)
    # print kth_slice,data.shape,indices
    return data[indices], data[kth_slice]
