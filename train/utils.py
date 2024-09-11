import os
import random
import numpy as np

import torch
import torch.distributed as dist


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def init_distributed_mode(dist_cfg):
    master_addr = os.getenv('MASTER_ADDR', default='localhost')
    master_port = os.getenv('MASTER_PORT', default='8888')
    method = f'tcp://{master_addr}:{master_port}'
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))  # global rank
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))

    ngpus_per_node = torch.cuda.device_count()
    dist_cfg.local_rank = rank % ngpus_per_node
    torch.cuda.set_device(dist_cfg.local_rank)

    dist.init_process_group(
        backend=dist_cfg.backend, init_method=method, world_size=world_size, rank=rank)

    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print