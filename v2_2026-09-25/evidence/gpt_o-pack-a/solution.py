"""Exact first-fit packing with indexed open-bin searches."""

from collections import deque
from heapq import heapify, heappop, heappush


def pack(lengths: list[int], params: dict) -> list[int]:
    capacity = params["capacity"]
    threshold = params["close_below"]
    max_open = params["max_open"]
    flush_every = params["flush_every"]

    # Leaves are bin IDs, so a leftmost qualifying leaf is the first fit.
    size = 1 << (max(1, len(lengths)) - 1).bit_length()
    missing = float("-inf")
    tree = [missing] * (2 * size)
    remaining = []
    oldest = deque()
    open_count = 0
    fullest = None  # Needed only once the memory guard actually fires.
    out = []

    def update(bin_id, value):
        node = size + bin_id
        tree[node] = value
        node >>= 1
        while node:
            left = tree[node * 2]
            right = tree[node * 2 + 1]
            value = left if left >= right else right
            if tree[node] == value:
                break
            tree[node] = value
            node >>= 1

    for i, length in enumerate(lengths):
        if length > capacity:
            out.append(len(remaining))
            remaining.append(None)
            continue

        if tree[1] >= length:
            node = 1
            while node < size:
                node *= 2
                if tree[node] < length:
                    node += 1
            placed = node - size
            free = remaining[placed] - length
        else:
            placed = len(remaining)
            remaining.append(capacity)
            oldest.append(placed)
            open_count += 1
            free = capacity - length
        out.append(placed)

        if free < threshold:
            remaining[placed] = None
            open_count -= 1
            update(placed, missing)
        else:
            remaining[placed] = free
            update(placed, free)
            if fullest is not None:
                heappush(fullest, (free, placed))

        if open_count > max_open:
            if fullest is None:
                fullest = [(r, b) for b, r in enumerate(remaining) if r is not None]
                heapify(fullest)
            while True:
                free, victim = heappop(fullest)
                if remaining[victim] == free:
                    break
            remaining[victim] = None
            open_count -= 1
            update(victim, missing)

        if (i + 1) % flush_every == 0 and open_count >= 2:
            while remaining[oldest[0]] is None:
                oldest.popleft()
            victim = oldest.popleft()
            remaining[victim] = None
            open_count -= 1
            update(victim, missing)

    return out
