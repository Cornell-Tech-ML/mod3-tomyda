# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA device execution."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for CUDA execution."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Apply a function element-wise to two tensors using CUDA."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Apply a reduction function to a tensor along a specified dimension."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on two tensors using CUDA."""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function.

    Args:
    ----
        fn: function mapping floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # saves indexes

        if i >= 0 and i < out_size:
            to_index(i, out_shape, out_index)  # retrieves out_idx for pos in storage
            broadcast_index(
                out_index, out_shape, in_shape, in_index
            )  # broadcasts out_index (bigger) to in_idx (smaller shape - input)
            in_storage_pos = index_to_position(
                in_index, in_strides
            )  # get position in storage of in_idx
            out[i] = fn(in_storage[in_storage_pos])
        else:
            return

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    """CUDA higher-order tensor zipWith (or map2) function.

    Args:
    ----
        fn: function mapping two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i >= 0 and i < out_size:
            to_index(i, out_shape, out_index)  # to get index of output
            broadcast_index(
                out_index, out_shape, a_shape, a_index
            )  # broadcasts big index (out shape) to smaller (in shape)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            in_storage_a_pos = index_to_position(a_index, a_strides)
            in_storage_b_pos = index_to_position(b_index, b_strides)

            out[i] = fn(a_storage[in_storage_a_pos], b_storage[in_storage_b_pos])
        else:
            return

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice sum kernel to prepare for reduce.

    Each block sums its portion of the array `a` and stores the result in `out`.

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int): length of `a`.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    if i >= size:
        cache[pos] = 0
    else:
        cache[pos] = a[i]
    cuda.syncthreads()

    if i < size:
        z = 2
        for _ in range(5):
            if (pos % z) == 0:
                cache[pos] += cache[pos + int(z / 2)]
                cuda.syncthreads()
            z *= 2
        if pos == 0:
            y = cuda.blockIdx.x
            out[y] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Perform a practice sum operation on a tensor using CUDA."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int, float], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function mapping two floats to a float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x

        # TODO: Implement for Task 3.3.

        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        cache[pos] = reduce_value

        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            pos2 = index_to_position(out_index, out_strides)
            idx_to_be_reduced = out_index[reduce_dim]
            global_idx = pos + idx_to_be_reduced * BLOCK_DIM

            out_index[reduce_dim] = global_idx

            if global_idx < a_shape[reduce_dim]:
                cache[pos] = a_storage[index_to_position(out_index, a_strides)]
                cuda.syncthreads()

                n = 1
                while n < BLOCK_DIM:
                    if pos % (2 * n) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + n])
                        cuda.syncthreads()
                    n *= 2
            if pos == 0:
                out[pos2] = cache[0]

        # TODO: Implement for Task 3.3.

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """_mm_practice is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    local_i = cuda.threadIdx.x
    local_k = cuda.threadIdx.y

    if local_i < size and local_k < size:
        out_pos = size * local_i + local_k
        a_shared[local_i, local_k] = a[out_pos]
        b_shared[local_i, local_k] = b[out_pos]
        cuda.syncthreads()
        tmp = 0.0
        idxx = 0
        while idxx < size:
            tmp = tmp + a_shared[local_i, idxx] * b_shared[idxx, local_k]
            out[out_pos] = tmp
            idxx += 1


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform a practice matrix multiplication on two tensors using CUDA."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:
    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Calculate batch strides
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Get batch index
    batch_idx = cuda.blockIdx.z

    # Define tile size
    TILE_SIZE = 32

    # Allocate shared memory
    shared_a = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)
    shared_b = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)

    # Compute global thread indices
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Compute local thread indices within the block
    local_row = cuda.threadIdx.x
    local_col = cuda.threadIdx.y

    # Initialize accumulator
    acc = 0.0

    # Loop over tiles
    for tile_idx in range(0, a_shape[2], TILE_SIZE):
        # Load data into shared memory from a_storage
        if row < a_shape[1] and (tile_idx + local_col) < a_shape[2]:
            a_index = (
                batch_idx * a_batch_stride
                + row * a_strides[1]
                + (tile_idx + local_col) * a_strides[2]
            )
            shared_a[local_row, local_col] = a_storage[a_index]
        else:
            shared_a[local_row, local_col] = 0.0

        # Load data into shared memory from b_storage
        if (tile_idx + local_row) < b_shape[1] and col < b_shape[2]:
            b_index = (
                batch_idx * b_batch_stride
                + (tile_idx + local_row) * b_strides[1]
                + col * b_strides[2]
            )
            shared_b[local_row, local_col] = b_storage[b_index]
        else:
            shared_b[local_row, local_col] = 0.0

        # Synchronize threads to ensure all data is loaded
        cuda.syncthreads()

        # Perform computation on the tile
        for k in range(TILE_SIZE):
            acc += shared_a[local_row, k] * shared_b[k, local_col]

        # Synchronize before loading next tile
        cuda.syncthreads()

    # Write the result to global memory
    if row < out_shape[1] and col < out_shape[2]:
        out_index = (
            batch_idx * out_strides[0] + row * out_strides[1] + col * out_strides[2]
        )
        out[out_index] = acc


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
