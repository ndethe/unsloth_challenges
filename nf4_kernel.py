import triton
import triton.language as tl
import torch
import math

@triton.jit
def RTNE(x):
    sign = x & 0x80000000
    exp = x & 0x7F800000
    man = x & 0x007FFFFF
    m16 = x & 0x00010000
    return tl.where(exp == 0,
                    sign >> 16, tl.where(exp == 255,
                                         tl.where(man == 0, 
                                                  x >> 16, (x >> 16) | 0x0040), tl.where(m16 == 0,
                                                   (x + 0x7FFF) >> 16, (x + 0x8000) >> 16)))

@triton.jit
def _optimized_dequantize_nf4_kernel(
    quant_ptr,
    absmax_ptr,      # uint8
    sup_absmax_ptr,  # float32
    code1_ptr,       # float32 (NF4 values)
    code2_ptr,       # float32 ((i - 127.)/128)
    offset_ptr,      # float32
    output_ptr,
    dt: tl.constexpr,
    absmax_numel: tl.constexpr,
    quant_numel: tl.constexpr,
    lbs0: tl.constexpr,
    lbs1: tl.constexpr,
    bs0: tl.constexpr,
    bs1: tl.constexpr
):
    pid0 = tl.program_id(0)
    
    supabsmax_offset = pid0
    absmax_offset = (pid0 << lbs1) + tl.arange(0, bs1).reshape(1, bs1, 1)
    quant_offset = (absmax_offset << lbs0) + tl.arange(0, bs0).reshape(bs0, 1, 1)
    out_offset = quant_offset << 1
    ileave = tl.arange(0,2)
    absmax_mask = absmax_offset < absmax_numel
    quant_mask = quant_offset < quant_numel
    
    offset = tl.load(offset_ptr)
    sup_absmax = tl.load(sup_absmax_ptr + supabsmax_offset, eviction_policy='evict_first')
    
    absmax_idx = tl.load(absmax_ptr + absmax_offset, mask=absmax_mask, eviction_policy='evict_first').to(tl.int32)
    base_absmax = tl.load(code2_ptr + absmax_idx, mask=absmax_mask, eviction_policy='evict_last')
    ## code2 and code1 are reused across different blocks so evict them last from cache.

    absmax = tl.fma(base_absmax, sup_absmax, offset) ## honestly this does not matter. only matters when there's downcasting after every operation
    
    packed = tl.load(quant_ptr + quant_offset, mask=quant_mask, eviction_policy='evict_first')
    bits = ((packed >> ((1-ileave) << 2)) & 0xF).to(tl.int32) 
    deq = tl.load(code1_ptr + bits, mask=quant_mask, eviction_policy='evict_last') * absmax

    if dt > 0:
        tl.store(output_ptr + out_offset + ileave, RTNE(tl.cast(deq, tl.uint32, bitcast=True)), mask=quant_mask)
    else:
        tl.store(output_ptr + out_offset + ileave, tl.cast(deq, tl.float16,  fp_downcast_rounding='rtne'), mask=quant_mask)
    
def _optimized_dequantize_nf4(weight, quant_state, output=None):
    """
    Optimized setup function with output reuse capability
    """
    m, n = weight.shape
    computations = m * n
    

    if output is None:
        # Reuse output tensor if provided to avoid allocation overhead
        if quant_state.dtype == torch.bfloat16:
            output = torch.empty(quant_state.shape, dtype=torch.uint16,
                                device=weight.device, requires_grad=False)
            dt = 1
        else:
            output = torch.empty(quant_state.shape, dtype=torch.float16,
                                device=weight.device, requires_grad=False)
            dt = -1
    else:
        if quant_state.dtype == torch.bfloat16:
            output = output.view(torch.uint16)
            dt = 1
        else:
            dt = -1
 
    # Launch kernel with the same grid as the original
    grid = (quant_state.state2.absmax.numel(),)
    _optimized_dequantize_nf4_kernel[grid](
        weight, 
        quant_state.absmax,
        quant_state.state2.absmax,
        quant_state.code,
        quant_state.state2.code,
        quant_state.offset,
        output,
        dt,
        quant_state.absmax.numel(),
        computations,
        math.floor(math.log2(quant_state.blocksize)) - 1,
        math.floor(math.log2(quant_state.state2.blocksize)),
        quant_state.blocksize // 2,
        quant_state.state2.blocksize
    )
    if quant_state.dtype == torch.bfloat16:
        output = output.view(torch.bfloat16)
    return output

def optimized_dequantize_nf4(weight, cached_output=None):
    """
    Entry point with option for output caching across calls
    """
    return _optimized_dequantize_nf4(weight.weight, weight.weight.quant_state, cached_output)