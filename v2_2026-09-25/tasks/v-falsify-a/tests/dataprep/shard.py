"""Static sharding of a document list across data-parallel ranks."""


def shard(n, world_size, rank, drop_last=False):
    """Document indices assigned to `rank`."""
    if drop_last:
        per = n // world_size
        return list(range(rank * per, (rank + 1) * per))
    per = (n + world_size - 1) // world_size
    idx = list(range(n)) + [n - 1] * (per * world_size - n)
    return idx[rank * per:(rank + 1) * per]
