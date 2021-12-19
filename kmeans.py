"""
k-means algorithm on GPU-accelerated cupy
"""
import numpy as np
import cupy as cp
from colour import Color
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Cluster data with GPU accelerated k-means algorithm.')
parser.add_argument("points", type=int, help="number of points")
parser.add_argument("clusters", type=int, help="number of clusters")
parser.add_argument("vectors", type=int, help="number of vector coordinates")
parser.add_argument("--iter", type=int, default=100,
                    help="max number of iterations")
parser.add_argument("--split", type=int, choices=range(1, 10),
                    help="number of initial splits to fit in GPU memory")
parser.add_argument("--save", type=str, help="output filename (.xyzrgb added)")
parser.add_argument("--gpu", type=int, default=1, help="gpu device")
parser.add_argument("--input", type=str, help="input data (.xyz format). Default = generate random data")

args = parser.parse_args()
PTS = args.points
CLUSTS = args.clusters
VECTS = args.vectors


def affect(X, centroids, gpu=0):
    """Attribute each point to a cluster, based on distances to centroids

    Args:
        X (np or cp array): Set of points
        centroids (np or cparray): Set of centroids
        gpu (int, optional): GPU device. Defaults to 0.

    Returns:
        [np or cp array]: array of cluster index for each point
    """
    with cp.cuda.Device(gpu):
        xp = cp.get_array_module(X)
        distances = xp.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        aff = xp.argmin(distances, axis=1)
        del distances
        cp.cuda.stream.get_current_stream().synchronize()
        return aff

def reseed(aff, n_clusters):
    """Change affectation if some clusters are empty

    Args:
        aff (array): affectation
        n_clusters (int): number of clusters

    Returns:
        [type]: fixed affectation
    """
    xp = cp.get_array_module(aff)
    uniques = xp.unique(aff)
    while len(uniques) != n_clusters:
        # reseed empty clusters
        uniques = cp.asnumpy(uniques)
        empty_clusters = np.setdiff1d(range(n_clusters), uniques, assume_unique=True) # empty clusters
        r = min(5, n_clusters//len(empty_clusters))
        print(f'reseed {len(empty_clusters)} clusters(x{r}): {empty_clusters}')
        new_idx = np.random.choice(n_clusters, len(empty_clusters)*r, replace=False)
        aff[new_idx] = np.tile(empty_clusters,r)
        uniques = xp.unique(aff)
        del empty_clusters
    del uniques
    return aff
    
def centers(X, aff, n_clusters, gpu=0):
    """Compute centroids based on points attributed to each cluster

    Args:
        X (np or cp array): Set of points
        aff (np or cp array): Array giving a cluster index for each point
        n_clusters (int): number of clusters
        gpu (int, optional): GPU device. Defaults to 0.

    Returns:
        [np or cp array]: sum of points
        [np or cp array]: number of points
    """
    with cp.cuda.Device(gpu):
        xp = cp.get_array_module(X)
        idx = xp.arange(n_clusters, dtype=int)
        mask = aff == idx[:, None]
        del idx
        sums = cp.where(mask[:, :, None], X, cp.float32(0)).sum(axis=1)
        counts = cp.count_nonzero(mask, axis=1).reshape((n_clusters, 1))
        del mask
        cp.cuda.stream.get_current_stream().synchronize()
        return sums, counts

def km(X, n_clusters, n_iters, gpu=1, split=3):
    """k-means algorithm

    Args:
        X (np or cp array): Set of points
        n_clusters (int): number of clusters
        n_iters (int): max number of iterations
        gpu (int, optional): GPU device. Defaults to 1.
        split (int, optional): Min number of data splits to fit GPU memory. Defaults to 3.

    Returns:
        [np or cp array]: centroids
        [np or cp array]: array of cluster index for each point
    """
    xp = cp.get_array_module(X)
    n_pts = len(X)
    n_split = split
    aff = None
    
    #set centroids as random selection among set of points
    indexes = xp.random.permutation(xp.arange(n_pts))[:n_clusters]
    centroids= X[indexes]
    del indexes

    for _ in tqdm(range(n_iters), desc=f'split {split:3}', colour='red'):
        # compute affectation
        for j in range(n_split):
            b_i = (j*n_pts)//n_split
            b_s = ((j+1)*n_pts)//n_split
            new_aff_n = affect(X[b_i:b_s, :], centroids, gpu)
            if j == 0:
                new_aff = new_aff_n
            else:
                new_aff = xp.concatenate([new_aff, new_aff_n])
            del new_aff_n
        if aff is not None and xp.all(new_aff == aff):
            del new_aff
            break
        aff = new_aff
        del new_aff
        aff = reseed(aff, n_clusters)

        # compute centroids
        counts = None
        for j in range(n_split):
            b_i = (j*n_pts)//n_split
            b_s = ((j+1)*n_pts)//n_split
            if j == 0:
                cents, cnts = centers(X[b_i:b_s, :],
                                        aff[b_i:b_s], n_clusters, gpu)
                centroids = cents
                counts = cnts
            else:
                cents, cnts = centers(X[b_i:b_s, :],
                                        aff[b_i:b_s], n_clusters, gpu)
                centroids += cents
                counts += cnts
        centroids = xp.true_divide(centroids, counts, dtype=xp.float32)
        cp.cuda.stream.get_current_stream().synchronize()
    return centroids, aff

def histo(counts):
    """print a horizontal histogram for cluster sizes

    Args:
        counts (array): number of clusters for each size
    """
    import shutil
    terminal_size = shutil.get_terminal_size()
    step = (np.max(counts) - np.min(counts))//int(terminal_size.lines*0.5-2)
    ct, be = np.histogram(counts, bins = np.arange(np.min(counts), np.max(counts)+step,step))
    mm = max(ct)

    for j in range(len(ct)):
        print(f'{be[j]:3}-{be[j+1]-1:3} [{ct[j]:4}] | {"█"*int(ct[j]/mm*0.6*(terminal_size.columns - 18))}')

with cp.cuda.Device(args.gpu):
    if args.input is None: # no input file, generate random points
        rng = cp.random.default_rng()
        vecteurs = rng.random((PTS, VECTS), dtype=cp.float32)
        del rng
        print(f'Generated {args.points} {args.vectors}-data at random')
    else:
        vecteurs = []
        with open(args.input, 'r', encoding='UTF-8') as file:
            while (line := file.readline()):
                vals = [float(token) for token in line.split()]
                vecteurs.append(vals)
        vecteurs = cp.array(vecteurs, cp.float32)
        args.points, args.vectors = vecteurs.shape
        print(f'Read {args.points} {args.vectors}-data from {args.input}')
    
    # increase split number until no out of memory error
    while True:
        try:
            centroids, aff = km(vecteurs, CLUSTS, args.iter,
                                gpu=args.gpu, split=args.split)
        except cp.cuda.memory.OutOfMemoryError:
            args.split += 1
        except KeyboardInterrupt:
            print('Ok, leaving and cooling down')
            exit()
        else:
            break

    cp.cuda.stream.get_current_stream().synchronize()

    aff = cp.asnumpy(aff)
    (unique, counts) = np.unique(aff, return_counts=True)
    if not (unique == list(range(CLUSTS))).all():
        print('Error, missing clusters')
    print(f'cluster size {np.min(counts)} to {np.max(counts)}')
    histo(counts)

# define color scale for clusters
black = Color('black')
white = Color('white')
scale = [c.rgb for c in black.range_to(white, CLUSTS)]

# save clusters as colors for each point
if args.save:
    with open(args.save+'.xyzrgb', 'w') as f:
        for i in range(PTS):
            print(
                f'{vecteurs[i][0]} {vecteurs[i][1]} {vecteurs[i][2]} {scale[aff[i]][0]} {scale[aff[i]][1]} {scale[aff[i]][2]}', file=f)
