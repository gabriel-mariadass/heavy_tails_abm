# OFC earthquake model: stress builds up then cascades (avalanche)

import heapq
import numpy as np


def simulate_ofc(L, alpha_ofc, n_events, seed=None, pbar=None):
    # Run OFC on L x L grid for n_events avalanches.
    # alpha_ofc = fraction of stress transferred to each neighbour.
    #
    # Speed: uses an offset accumulator so the loading phase costs O(1)
    # instead of O(L^2 log L^2).  F stores F_stored = F_true - offset;
    # a global scalar offset is incremented each loading step so all
    # true stresses rise by delta without touching the array or heap.
    if not (0 < alpha_ofc <= 0.25):
        raise ValueError(f"alpha_ofc={alpha_ofc} must be in (0, 0.25].")

    F_th = 1.0
    rng = np.random.default_rng(seed)

    # Initialise: F_stored = F_true (offset=0)
    F = rng.uniform(0.0, F_th, size=(L, L))
    offset = 0.0

    def get_neighbours(r, c):
        nbrs = []
        if r > 0:
            nbrs.append((r - 1, c))
        if r < L - 1:
            nbrs.append((r + 1, c))
        if c > 0:
            nbrs.append((r, c - 1))
        if c < L - 1:
            nbrs.append((r, c + 1))
        return nbrs

    neighbours = [[get_neighbours(r, c) for c in range(L)] for r in range(L)]

    sizes = np.empty(n_events, dtype=np.int64)

    # Lazy-deletion heap stores (-F_stored, r, c, version).
    version = np.zeros((L, L), dtype=np.int64)
    heap = []
    for r in range(L):
        for c in range(L):
            heapq.heappush(heap, (-F[r, c], r, c, 0))

    def push(r, c):
        version[r, c] += 1
        heapq.heappush(heap, (-F[r, c], r, c, version[r, c]))

    def pop_valid():
        while True:
            neg_f, r, c, ver = heapq.heappop(heap)
            if ver == version[r, c]:
                return r, c, -neg_f  # returns f_stored

    for event in range(n_events):
        # Loading phase: advance offset so F_true_max reaches F_th.
        # Because offset shifts all cells equally, heap order is preserved —
        # no rebuild needed.
        r_max, c_max, f_stored_max = pop_valid()
        push(r_max, c_max)  # put back unchanged

        delta = F_th - (f_stored_max + offset)
        offset += delta

        # Cascade phase: fire all cells with F_true >= F_th,
        # i.e. F_stored >= F_th - offset.
        threshold_stored = F_th - offset
        firing_queue = [(r_max, c_max)]

        size = 0
        while firing_queue:
            r, c = firing_queue.pop()
            if F[r, c] < threshold_stored:
                continue
            f_true_before = F[r, c] + offset
            F[r, c] = -offset          # set F_true = 0
            size += 1
            push(r, c)

            for (nr, nc) in neighbours[r][c]:
                F[nr, nc] += alpha_ofc * f_true_before
                push(nr, nc)
                if F[nr, nc] >= threshold_stored:
                    firing_queue.append((nr, nc))

        sizes[event] = size
        if pbar is not None:
            pbar.update(1)

    return sizes
