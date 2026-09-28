from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer(times_list: list, device: torch.device):
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        yield  # Execute the block of code inside the context manager

        end.record()

    torch.cuda.synchronize(device)
    times_list.append(start.elapsed_time(end))
