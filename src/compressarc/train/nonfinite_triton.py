"""Direct, read-only NaN/Inf counts for lists of contiguous CUDA tensors."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scan(META, PARTIAL, DTYPE:tl.constexpr, BLOCK:tl.constexpr):
    tile = tl.program_id(0)
    address = tl.load(META+tile*3)
    offset = tl.load(META+tile*3+1)
    length = tl.load(META+tile*3+2)
    pointer = address.to(tl.pointer_type(DTYPE))
    index = tl.arange(0, BLOCK)
    values = tl.load(pointer+offset+index, index<length, other=0.)
    # Count elements, not just a boolean flag. +/-Inf both count as infinity.
    nan_count = tl.sum(((values != values)&(index<length)).to(tl.int64),0)
    inf_count = tl.sum(((tl.abs(values)==float('inf'))&(index<length)).to(tl.int64),0)
    tl.store(PARTIAL+tile*2,nan_count)
    tl.store(PARTIAL+tile*2+1,inf_count)


DTYPES = {torch.float16:tl.float16,torch.bfloat16:tl.bfloat16,
          torch.float32:tl.float32,torch.float64:tl.float64}


def count_tiles(metadata, dtype):
    """Metadata owns no tensor storage; caller keeps source tensors alive."""
    partial = torch.empty((metadata.shape[0],2),device=metadata.device,dtype=torch.int64)
    with torch.cuda.device(metadata.device):
        _scan[(metadata.shape[0],)](metadata,partial,DTYPE=DTYPES[dtype],BLOCK=1024)
        return partial.sum(dim=0)
