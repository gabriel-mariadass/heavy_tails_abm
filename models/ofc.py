# OFC earthquake model: stress builds up then cascades (avalanche)

import heapq
import numpy as np


def simulate_ofc(L, alpha_ofc, n_events, seed=None):
    # Run OFC on L x L grid for n_events avalanches
    # alpha_ofc = fraction of stress transferred to each neighbour
    if not (0 < alpha_ofc <= 0.25):
        raise ValueError(f"alpha_ofc={alpha_ofc} must be in (0, 0.25].")

    F_th = 1.0
    rng = np.random.default_rng(seed)

    # random initial stress
    F = rng.uniform(0.0, F_th, size=(L, L))

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

    # lazy-deletion heap: (-stress, r, c, version)
    version = np.zeros((L, L), dtype=np.int64)
    heap = []
    for r in range(L):
        for c in range(L):
            heapq.heappush(heap, (-F[r, c], r, c, 0))

    def push(r, c):
        # push updated cell with new version to invalidate old entry
        version[r, c] += 1
        heapq.heappush(heap, (-F[r, c], r, c, version[r, c]))

    def pop_valid():
        # skip stale heap entries
        while True:
            neg_f, r, c, ver = heapq.heappop(heap)
            if ver == version[r, c]:
                return r, c, -neg_f

    for event in range(n_events):
        # loading phase: shift all stress so max cell hits threshold
        r_max, c_max, f_max = pop_valid()
        push(r_max, c_max)

        delta = F_th - f_max
        F += delta

        # rebuild heap after global shift (all entries are stale)
        heap = []
        for r in range(L):
            for c in range(L):
                version[r, c] += 1
                heapq.heappush(heap, (-F[r, c], r, c, version[r, c]))

        # cascade phase: fire cells above threshold
        firing_queue = []
        r_fire, c_fire, _ = pop_valid()
        firing_queue.append((r_fire, c_fire))

        size = 0
        while firing_queue:
            r, c = firing_queue.pop()
            if F[r, c] < F_th:
                # already relaxed earlier in this cascade
                continue
            f_before = F[r, c]
            F[r, c] = 0.0
            size += 1
            push(r, c)

            for (nr, nc) in neighbours[r][c]:
                F[nr, nc] += alpha_ofc * f_before
                push(nr, nc)
                if F[nr, nc] >= F_th:
                    firing_queue.append((nr, nc))

        sizes[event] = size

    return sizes
