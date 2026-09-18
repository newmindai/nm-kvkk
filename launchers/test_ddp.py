"""4-GPU sanity test: NCCL all_reduce + environment report. Run under torchrun."""

import os

import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", -1))
assert local_rank >= 0, "run me with torchrun"

dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
rank, world = dist.get_rank(), dist.get_world_size()

t = torch.ones(1, device=f"cuda:{local_rank}")
dist.all_reduce(t)

print(
    f"rank {rank}/{world} | {torch.cuda.get_device_name(local_rank)} | "
    f"all_reduce sum = {t.item():.0f} (expect {world})",
    flush=True,
)

if rank == 0:
    import transformers

    import gliner2  # resolved via PYTHONPATH -> the project clone

    print(
        f"torch {torch.__version__} | cuda {torch.version.cuda} | "
        f"transformers {transformers.__version__} | gliner2 from {gliner2.__file__}",
        flush=True,
    )

dist.destroy_process_group()
