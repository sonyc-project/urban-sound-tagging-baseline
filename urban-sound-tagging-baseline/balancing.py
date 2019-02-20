# Copy from https://github.com/sonyc-project/sensor-embedding/blob/master/sensor_embedding/downstream/balancing.py
import numpy as np
from sklearn.neighbors import NearestNeighbors


def calculate_imbalance_ratio(y):
    label_counts = np.zeros((y.shape[1]), dtype='float32')
    for labelset in y:
        label_counts += (labelset > 0).astype(int)

    irlbl = compute_irlbl(label_counts)

    return irlbl, label_counts


def compute_irlbl(label_counts):
    return label_counts.max() / label_counts


def get_all_instances_of_label(label, y):
    inst_idxs = []
    for ex_idx, labelset in enumerate(y):
        if labelset[label] > 0:
            inst_idxs.append(ex_idx)

    return np.array(inst_idxs)


def mlsmote(X, y, k_neighbors=5, max_iters_per_class=None, oversample_iters=1, thresh_type="mean"):
    X_new = X.copy()
    y_new = y.copy()

    L = y.shape[1]
    knn = NearestNeighbors(n_neighbors=k_neighbors)

    for _ in range(oversample_iters):
        irlbl, label_counts = calculate_imbalance_ratio(y)
        thresh = compute_thresh(irlbl, thresh_type)
        for label in range(L):
            irlbl = compute_irlbl(label_counts)
            min_bag_idxs = get_all_instances_of_label(label, y)

            n_iters = 0
            while irlbl[label] > thresh and (max_iters_per_class is None or n_iters < max_iters_per_class):
                for list_idx, sample_idx in enumerate(min_bag_idxs):
                    x_seed = X[sample_idx]
                    y_seed = y[sample_idx]
                    sample_idxs = np.concatenate((min_bag_idxs[:list_idx],
                                                  min_bag_idxs[list_idx + 1:]))
                    X_knn = X[sample_idxs]
                    # Use mean for each nearest neighbors
                    X_knn = X_knn.mean(axis=1)
                    knn.fit(X_knn)
                    neighbors = knn.kneighbors(x_seed.mean(axis=0).reshape(1, x_seed.shape[-1]),
                                               return_distance=False)[0]
                    ref_neigh_idx = sample_idxs[np.random.choice(neighbors)]
                    x_neigh = X[ref_neigh_idx]
                    y_neigh = y[ref_neigh_idx]

                    x_synth, y_synth = mlsmote_new_sample(x_seed, y_seed,
                                                          x_neigh, y_neigh)

                    # Append new samples to the dataset
                    X_new = np.vstack((X_new, [x_synth]))
                    y_new = np.vstack((y_new, [y_synth]))

                    # Update label counts
                    label_counts += (y_synth > 0).astype(int)

                    # Update IRLbl
                    irlbl = compute_irlbl(label_counts)
                    if irlbl[label] <= thresh:
                        break
                n_iters += 1

    return X_new, y_new, label_counts


# Labelset SMOTE
def calculate_imbalance_ratio_ls(y_ls, labelsets):
    labelset_counts = np.zeros((len(labelsets)), dtype='float32')
    for yi in y_ls:
        labelset_counts[yi] += 1

    irlbl = compute_irlbl(labelset_counts)

    return irlbl, labelset_counts


def get_all_instances_of_subset_labelset(labelset_idx, y_ls, labelsets):
    target_labelset = labelsets[labelset_idx]

    inst_idxs = []
    for idx, yi in enumerate(y_ls):
        if labelsets[yi].issubset(target_labelset):
            inst_idxs.append(idx)

    return np.array(inst_idxs)


def get_compatible_ref_instances(labelset_idx, seed_labelset_idx, inst_idxs, y_ls, labelsets, seed_idx=None):
    target_labelset = labelsets[labelset_idx]
    seed_labelset = labelsets[seed_labelset_idx]

    ref_idxs = []
    for idx in inst_idxs:
        if seed_idx is not None and seed_idx == idx:
            continue

        if labelsets[y_ls[idx]].union(seed_labelset) == target_labelset:
            ref_idxs.append(idx)

    return np.array(ref_idxs)


def compute_labelsets(y):
    labelsets = set()

    for yi in y:
        labelsets.add(frozenset(np.nonzero(yi > 0)[0]))

    labelsets = list(labelsets)
    labelset_to_idx = {x: idx for idx, x in enumerate(labelsets)}

    y_ls = []
    for yi in y:
        y_ls.append(labelset_to_idx[frozenset(np.nonzero(yi > 0)[0])])
    y_ls = np.array(y_ls)

    return y_ls, labelsets, labelset_to_idx


def lssmote(X, y, k_neighbors=5, max_iters_per_labelset=None, oversample_iters=1, thresh_type="mean"):
    X_new = X.copy()
    y_new = y.copy()

    y_ls, labelsets, labelset_to_idx = compute_labelsets(y_new)

    L = len(labelsets)
    knn = NearestNeighbors(n_neighbors=k_neighbors)

    for _ in range(oversample_iters):
        irlbl, labelset_counts = calculate_imbalance_ratio_ls(y_ls, labelsets)
        thresh = compute_thresh(irlbl, thresh_type)

        for labelset_idx in range(L):
            irlbl = compute_irlbl(labelset_counts)

            min_bag_idxs = get_all_instances_of_subset_labelset(labelset_idx, y_ls, labelsets)
            n_iters = 0
            while irlbl[labelset_idx] > thresh and (max_iters_per_labelset is None or n_iters < max_iters_per_labelset):
                for list_idx, sample_idx in enumerate(min_bag_idxs):
                    x_seed = X[sample_idx]
                    y_seed = y[sample_idx]
                    y_seed_labelset = y_ls[sample_idx]

                    sample_idxs = get_compatible_ref_instances(labelset_idx, y_seed_labelset, min_bag_idxs, y_ls,
                                                               labelsets, seed_idx=list_idx)

                    X_knn = X[sample_idxs]
                    # Use mean for each nearest neighbors
                    X_knn = X_knn.mean(axis=1)
                    knn.fit(X_knn)
                    neighbors = knn.kneighbors(x_seed.mean(axis=0).reshape(1, x_seed.shape[-1]),
                                               n_neighbors=min(len(sample_idxs), k_neighbors),
                                               return_distance=False)[0]
                    ref_neigh_idx = sample_idxs[np.random.choice(neighbors)]
                    x_neigh = X[ref_neigh_idx]
                    y_neigh = y[ref_neigh_idx]

                    x_synth, y_synth = mlsmote_new_sample(x_seed, y_seed,
                                                          x_neigh, y_neigh)

                    # Append new samples to the dataset
                    X_new = np.vstack((X_new, [x_synth]))
                    y_new = np.vstack((y_new, [y_synth]))

                    assert labelset_to_idx[frozenset(np.nonzero(y_synth > 0)[0])] == labelset_idx

                    # Update label counts
                    labelset_counts[labelset_idx] += 1

                    # Update IRLbl
                    irlbl = compute_irlbl(labelset_counts)
                    if irlbl[labelset_idx] <= thresh:
                        break

                n_iters += 1

    return X_new, y_new, labelset_counts


def compute_thresh(irlbl, thresh_type):
    if thresh_type == "mean":
        return irlbl.mean()
    elif thresh_type.startswith("percentile_"):
        perc = float(thresh_type.split('_')[1])
        return np.percentile(irlbl, perc)
    else:
        raise ValueError("Invalid threshold type: {}".format(thresh_type))


def mlsmote_new_sample(x_seed, y_seed, x_neigh, y_neigh):
    alpha = np.random.random()
    x_synth = alpha * x_seed + (1 - alpha) * x_neigh
    y_synth = alpha * y_seed + (1 - alpha) * y_neigh
    return x_synth, y_synth