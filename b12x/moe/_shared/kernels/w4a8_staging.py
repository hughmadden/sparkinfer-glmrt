"""Exact-width views of the N256/K128 lane-major W4A8 resident layout.

These staging primitives do not change the resident representation. Callers
provide valid, 32-row-aligned slices and synchronize cp.async before reading
shared memory. Slice width is static geometry; position and strides are runtime.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from b12x._lib.intrinsics import cp_async4_shared_global, get_ptr_as_int64, st_shared_u32


@cute.jit
def stage_repacked_b_slice(
    source: cute.Tensor,
    shared_base: Int32,
    expert_word_base: Int64,
    k_tiles: Int32,
    k_tile: Int32,
    n_start: Int32,
    thread: Int32,
    threads: cutlass.Constexpr,
    width: cutlass.Constexpr,
):
    """Copy N64/N128/N192 weights into compact lane-major shared storage.

    The compact destination has shape [4,width/32,32,4] in u32 words.
    A slice may cross an N256 boundary, including independently padded W13
    projection halves. All source offsets and products use 64-bit arithmetic.
    """
    assert width in (64, 128, 192)
    chunks = width // 32
    transfers = 4 * chunks * 32
    for iteration in cutlass.range_constexpr((transfers + threads - 1) // threads):
        index = thread + Int32(iteration * threads)
        if index < Int32(transfers):
            lane = index % Int32(32)
            chunk = (index // Int32(32)) % Int32(chunks)
            kb = index // Int32(32 * chunks)
            n = Int64(n_start) + Int64(chunk) * Int64(32)
            tile = (n // Int64(256)) * Int64(k_tiles) + Int64(k_tile)
            word = (expert_word_base + tile * Int64(4096)
                    + Int64(kb) * Int64(1024)
                    + ((n % Int64(256)) // Int64(32)) * Int64(128)
                    + Int64(lane) * Int64(4))
            cp_async4_shared_global(shared_base + index * Int32(16),
                                    get_ptr_as_int64(source, word))


@cute.jit
def stage_repacked_sfb_slice(
    source: cute.Tensor,
    shared_base: Int32,
    expert_word_base: Int64,
    k_tiles: Int32,
    k_tile: Int32,
    n_start: Int32,
    thread: Int32,
    threads: cutlass.Constexpr,
    width: cutlass.Constexpr,
):
    """Copy the corresponding N rows of four packed K32 UE8M0 scales."""
    assert width in (64, 128, 192)
    transfers = width // 4
    for iteration in cutlass.range_constexpr((transfers + threads - 1) // threads):
        index = thread + Int32(iteration * threads)
        if index < Int32(transfers):
            n = Int64(n_start) + Int64(index) * Int64(4)
            tile = (n // Int64(256)) * Int64(k_tiles) + Int64(k_tile)
            word = expert_word_base + tile * Int64(256) + n % Int64(256)
            cp_async4_shared_global(shared_base + index * Int32(16),
                                    get_ptr_as_int64(source, word))


@cute.jit
def stage_repacked_b_k_slice(
    source: cute.Tensor,
    shared_base: Int32,
    expert_word_base: Int64,
    k_tiles: Int32,
    k_start: Int32,
    n_start: Int32,
    thread: Int32,
    threads: cutlass.Constexpr,
    depth: cutlass.Constexpr,
):
    """Copy N128 by K64/K128/K192 into [depth/32,4,32,4] u32.

    Both starts must be aligned to 32. Handles crossings of both packed tile
    axes; the caller owns bounds and completion just as for N-slice staging.
    """
    assert depth in (64, 128, 192)
    transfers = (depth // 32) * 4 * 32
    for iteration in cutlass.range_constexpr((transfers + threads - 1) // threads):
        index = thread + Int32(iteration * threads)
        if index < Int32(transfers):
            lane = index % Int32(32)
            chunk = (index // Int32(32)) % Int32(4)
            kb = index // Int32(128)
            n = Int64(n_start) + Int64(chunk) * Int64(32)
            k = Int64(k_start) // Int64(32) + Int64(kb)
            tile = (n // Int64(256)) * Int64(k_tiles) + k // Int64(4)
            word = (expert_word_base + tile * Int64(4096)
                    + (k % Int64(4)) * Int64(1024)
                    + ((n % Int64(256)) // Int64(32)) * Int64(128)
                    + Int64(lane) * Int64(4))
            cp_async4_shared_global(shared_base + index * Int32(16),
                                    get_ptr_as_int64(source, word))


@cute.jit
def stage_repacked_sfb_k_slice(
    source: cute.Tensor,
    destination: cute.Tensor,
    expert_word_base: Int64,
    k_tiles: Int32,
    k_start: Int32,
    n_start: Int32,
    thread: Int32,
    threads: cutlass.Constexpr,
    depth: cutlass.Constexpr,
):
    """Gather N128 scales as [ceil(depth/128),128] packed u32 words.

    Only the final scale word is zero-padded to four bytes for MMA byte-id
    selection. Weight payload has no padding. Stores are synchronous; callers
    synchronize threads as well as completing the weight cp.async group.
    """
    assert depth in (64, 128, 192)
    groups = (depth + 127) // 128
    for iteration in cutlass.range_constexpr((128 * groups + threads - 1) // threads):
        index = thread + Int32(iteration * threads)
        if index < Int32(128 * groups):
            row = index % Int32(128)
            group = index // Int32(128)
            n = Int64(n_start) + Int64(row)
            packed = cutlass.Uint32(0)
            for byte in cutlass.range_constexpr(4):
                block = group * Int32(4) + Int32(byte)
                if block < Int32(depth // 32):
                    k = Int64(k_start) // Int64(32) + Int64(block)
                    tile = (n // Int64(256)) * Int64(k_tiles) + k // Int64(4)
                    word = source[expert_word_base + tile * Int64(256) + n % Int64(256)]
                    value = (cutlass.Uint32(word) >> cutlass.Uint32((k % Int64(4)) * Int64(8))) & cutlass.Uint32(255)
                    packed |= value << cutlass.Uint32(byte * 8)
            destination[index] = packed.to(destination.element_type)


@cute.jit
def stage_v41_exact_b(source: cute.Tensor, shared_base: Int32,
                      expert_word_base: Int64, n_start: Int32, k_start: Int32,
                      thread: Int32, width: cutlass.Constexpr,
                      intermediate: cutlass.Constexpr, hidden: cutlass.Constexpr,
                      gated: cutlass.Constexpr, projection: cutlass.Constexpr = 0):
    """Exact TP4 payload: FC1 N128 tiles (last N64), FC2 K128 tiles (last K64).

    Shared payload remains the existing compact MMA layout. Out-of-range tails
    are zeroed only in shared MMA scratch, never in resident weights.
    FC1 k_start is a K128 tile; FC2 k_start is a logical K row.
    """
    chunks = width // 32 if gated else 4
    blocks = 4 if gated else width // 32
    for iteration in cutlass.range_constexpr((blocks * chunks * 32 + 127) // 128):
        index = thread + Int32(iteration * 128)
        if index < Int32(blocks * chunks * 32):
            lane = index % Int32(32)
            chunk = (index // Int32(32)) % Int32(chunks)
            kb = index // Int32(32 * chunks)
            if cutlass.const_expr(gated):
                n = n_start + chunk * Int32(32)
                tile = n // Int32(128)
                rows = Int32(128)
                if tile == Int32(intermediate // 128):
                    rows = Int32(intermediate % 128)
                word = (expert_word_base + Int64(projection * intermediate * (hidden // 8))
                        + Int64(tile * 128 * (hidden // 8))
                        + Int64((k_start * 4 + kb) * (rows // 32) * 128)
                        + Int64((n % 128 // 32) * 128 + lane * 4))
            else:
                n = n_start + chunk * Int32(32)
                k = k_start // Int32(32) + kb
                word = (expert_word_base + Int64((n // 128) * 128 * (intermediate // 8))
                        + Int64(k * 512 + (n % 128 // 32) * 128 + lane * 4))
            valid = n < Int32(intermediate) if cutlass.const_expr(gated) else k < Int32(intermediate // 32)
            address = shared_base + index * Int32(16)
            if cutlass.const_expr(intermediate % width == 0):
                # Callers launch ceil(I/width) slices. A whole-width slice
                # never crosses I; only width 128 needs shared tail zeros.
                cp_async4_shared_global(address, get_ptr_as_int64(source, word))
            else:
                if valid:
                    cp_async4_shared_global(address, get_ptr_as_int64(source, word))
                else:
                    for element in cutlass.range_constexpr(4):
                        st_shared_u32(address + Int32(element * 4), cutlass.Uint32(0))


@cute.jit
def stage_v41_exact_sfb(source: cute.Tensor, destination,
                        expert_word_base: Int64, n_start: Int32, k_start: Int32,
                        thread: Int32, width: cutlass.Constexpr,
                        intermediate: cutlass.Constexpr, hidden: cutlass.Constexpr,
                        gated: cutlass.Constexpr, projection: cutlass.Constexpr = 0):
    """Gather exact resident scales into the original shared MMA scale layout."""
    if cutlass.const_expr(gated):
        if thread < Int32(width // 4):
            n = n_start + thread * Int32(4)
            tile = n // Int32(128)
            count = Int32(128)
            if tile == Int32(intermediate // 128):
                count = Int32(intermediate % 128)
            word = (expert_word_base + Int64(projection) * Int64(intermediate * hidden // 128)
                    + Int64(tile) * Int64(hidden)
                    + Int64(k_start * count + n % 128))
            address = destination + thread * Int32(16)
            if cutlass.const_expr(intermediate % width == 0):
                cp_async4_shared_global(address, get_ptr_as_int64(source, word))
            else:
                if n < Int32(intermediate):
                    cp_async4_shared_global(address, get_ptr_as_int64(source, word))
                else:
                    for element in cutlass.range_constexpr(4):
                        st_shared_u32(address + Int32(element * 4), cutlass.Uint32(0))
    else:
        assert intermediate == 576 and width in (64, 128, 192)
        source_halves = cute.recast_tensor(source, cutlass.Uint16)
        # FC2's n_start is N128-aligned and all 128 threads own one row.
        # Every emitted scale group begins before K=576, including width 128.
        row = thread
        tile_base = expert_word_base + Int64(n_start // 128) * Int64(576)
        groups = (width + 127) // 128
        for group in cutlass.range_constexpr(groups):
            k = k_start // Int32(32) + Int32(group * 4)
            tile = k // Int32(4)
            phase = k % Int32(4)
            word = cutlass.Uint32(0)
            if cutlass.const_expr(width == 192 and group == 0):
                # The first four-byte groups begin at K32 blocks 0/6/12;
                # none touches the two-byte resident tail at block 16.
                word = cutlass.Uint32(source[tile_base + Int64(tile * 128 + row)])
            else:
                if tile == Int32(4):
                    word = cutlass.Uint32(source_halves[tile_base * Int64(2)
                        + Int64(1024 + row)])
                else:
                    word = cutlass.Uint32(source[tile_base + Int64(tile * 128 + row)])
            packed = word >> cutlass.Uint32(phase * 8)
            if cutlass.const_expr(width == 192 and group == 0):
                if phase > Int32(0):
                    next_word = cutlass.Uint32(source[tile_base + Int64((tile + 1) * 128 + row)])
                    packed |= next_word << cutlass.Uint32((4 - phase) * 8)
            if cutlass.const_expr(min(4, width // 32 - group * 4) == 2):
                packed &= cutlass.Uint32(65535)
            destination[thread + Int32(group * 128)] = packed.to(destination.element_type)
