"""Numerical checks with an optional pointer-table CUDA fast path."""
from contextlib import nullcontext
import torch


SUPPORTED = {torch.float16,torch.bfloat16,torch.float32,torch.float64}


def tile_metadata(tensors, block=1024):
    """A row is (tensor address, element offset, valid element count).

    Do not deduplicate aliases: the original concatenation counts each occurrence.
    data_ptr includes storage_offset. No detach/reshape views are needed.
    """
    rows = []
    for tensor in tensors:
        pointer,size = tensor.data_ptr(),tensor.numel()
        rows.extend((pointer,start,min(block,size-start)) for start in range(0,size,block))
    return rows


class NonfiniteCounter:
    def __init__(self, implementation='reference'):
        if implementation not in ('reference','triton'):
            raise ValueError('training.diagnostics_implementation must be reference or triton')
        self.implementation = implementation
        self.cache = {}

    def count(self, tensors, category, scope=None):
        scope = scope or (lambda name:nullcontext())
        reference,fast = {},{}
        with scope('prepare'):
            for value in tensors:
                if value is None:
                    continue
                if self.implementation == 'reference':
                    if value.is_floating_point() or value.is_complex():
                        reference.setdefault(value.device,[]).append(value.detach().reshape(-1))
                else:
                    # dtype/device metadata do not dispatch detach/reshape per parameter.
                    dtype = value.dtype
                    if not (dtype.is_floating_point or dtype.is_complex):
                        continue
                    if value.is_cuda and dtype in SUPPORTED and value.is_contiguous():
                        if value.numel():
                            fast.setdefault((value.device,dtype),[]).append(value)
                    else:
                        # Preserve semantics for complex values, noncontiguous views,
                        # and CPU tensors via the original implementation.
                        reference.setdefault(value.device,[]).append(value.detach().reshape(-1))
        nan_count = inf_count = 0
        for values in reference.values():
            with scope('concatenate'):
                flattened = torch.cat(values)
            with scope('reduce'):
                counts = torch.stack((torch.isnan(flattened).sum(),torch.isinf(flattened).sum()))
            with scope('transfer'):
                counts = counts.cpu().tolist()
            nan_count += counts[0]
            inf_count += counts[1]
        if fast:
            from .nonfinite_triton import count_tiles
        for (device,dtype),values in fast.items():
            with scope('metadata'):
                # Addresses change when zero_grad replaces grad buffers or a
                # parameter changes storage. Rebuild instead of reading stale data.
                signature = tuple((id(v),v.data_ptr(),v.numel()) for v in values)
                key = (category,device,dtype)
                cached = self.cache.get(key)
                if cached is None or cached[0] != signature:
                    metadata = torch.tensor(tile_metadata(values),dtype=torch.int64,device=device)
                    self.cache[key] = (signature,metadata)
                else:
                    metadata = cached[1]
            with scope('reduce'):
                counts = count_tiles(metadata,dtype)
            with scope('transfer'):
                counts = counts.cpu().tolist()
            nan_count += counts[0]
            inf_count += counts[1]
        return nan_count,inf_count
