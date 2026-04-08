# Cont-Bouchaud model: herding agents on a grid -> fat tail returns

import numpy as np
from scipy.ndimage import label


def simulate_cb(L, p, a, n_steps, lam=1.0, seed=None):
    # Run CB model for n_steps on L x L grid
    # p = bond probability, a = activity prob per cluster
    if 2 * a > 1:
        raise ValueError(f"Activity probability 2a={2*a} must be <= 1.")

    rng = np.random.default_rng(seed)
    N = L * L
    returns = np.empty(n_steps, dtype=np.float64)

    for t in range(n_steps):
        # draw horizontal and vertical bonds
        h_bonds = rng.random((L, L - 1)) < p
        v_bonds = rng.random((L - 1, L)) < p

        # build grid where agent cells = 1, bond cells = 1 if connected
        grid = np.zeros((2 * L - 1, 2 * L - 1), dtype=np.int8)
        grid[0::2, 0::2] = 1
        grid[0::2, 1::2] = h_bonds.astype(np.int8)
        grid[1::2, 0::2] = v_bonds.astype(np.int8)

        labeled, n_clusters = label(grid)

        # get cluster label for each agent
        agent_labels = labeled[0::2, 0::2].ravel()

        # each cluster buys, sells, or holds
        max_label = n_clusters
        u = rng.random(max_label + 1)
        direction = np.zeros(max_label + 1, dtype=np.float64)
        direction[u < a] = 1.0
        direction[(u >= a) & (u < 2 * a)] = -1.0

        # return = sum of signed cluster volumes
        sizes = np.bincount(agent_labels, minlength=max_label + 1).astype(np.float64)
        signed_volumes = sizes * direction
        returns[t] = lam * signed_volumes.sum()

    return returns
