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

    @cuda.jit()
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert linear index to multidimensional index
        to_index(i, out_shape, out_index)

        # Handle broadcasting
        broadcast_index(out_index, out_shape, in_shape, in_index)

        # Compute positions
        out_pos = index_to_position(out_index, out_strides)
        in_pos = index_to_position(in_index, in_strides)

        # Apply the function
        out[out_pos] = fn(in_storage[in_pos])

    return _map  # type: ignore


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

    @cuda.jit()
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
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert linear index to multidimensional index
        to_index(i, out_shape, out_index)

        # Handle broadcasting for both inputs
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        # Compute positions
        out_pos = index_to_position(out_index, out_strides)
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)

        # Apply the function
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice sum kernel to prepare for reduce.

    Each block sums its portion of the array `a` and stores the result in `out`.

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int): length of `a`.

    """
    BLOCK_DIM = THREADS_PER_BLOCK

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    tid = cuda.threadIdx.x

    # Load data into shared memory
    if i < size:
        cache[tid] = a[i]
    else:
        cache[tid] = 0.0

    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            cache[tid] += cache[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Write result to output
    if tid == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Perform a practice sum operation on a tensor using CUDA."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    out_size = blockspergrid
    out = TensorData([0.0 for _ in range(out_size)], (out_size,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out._storage, a._tensor._storage, size
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

    @cuda.jit()
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
        BLOCK_DIM = cuda.blockDim.x
        pos = cuda.threadIdx.x
        out_pos = cuda.blockIdx.x

        if out_pos >= out_size:
            return

        # Initialize index arrays
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert linear index to multidimensional index
        to_index(out_pos, out_shape, out_index)

        # Initialize accumulator
        total = reduce_value

        # Get the size along the reduction dimension
        reduce_dim_size = a_shape[reduce_dim]

        # Iterate over the reduction dimension in steps of BLOCK_DIM
        for s in range(pos, reduce_dim_size, BLOCK_DIM):
            # Copy out_index to in_index
            for d in range(len(out_shape)):
                in_index[d] = out_index[d]
            in_index[reduce_dim] = s

            # Compute input position
            in_pos = index_to_position(in_index, a_strides)

            # Accumulate the reduction
            total = fn(total, a_storage[in_pos])

        # Allocate shared memory for partial results
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        cache[pos] = total
        cuda.syncthreads()

        # Perform parallel reduction within the block
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        # Write the result to the output tensor
        if pos == 0:
            out_position = index_to_position(out_index, out_strides)
            out[out_position] = cache[0]

    return _reduce  # type: ignore


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
    # Shared memory arrays (maximum size 32x32)
    a_shared = cuda.shared.array((32, 32), numba.float64)
    b_shared = cuda.shared.array((32, 32), numba.float64)

    tx = cuda.threadIdx.x  # Row index
    ty = cuda.threadIdx.y  # Column index

    # Load elements into shared memory
    if tx < size and ty < size:
        a_idx = tx * size + ty
        b_idx = tx * size + ty
        a_shared[tx, ty] = a[a_idx]
        b_shared[tx, ty] = b[b_idx]
    else:
        a_shared[tx, ty] = 0.0
        b_shared[tx, ty] = 0.0

    # Synchronize threads after loading data
    cuda.syncthreads()

    # Perform matrix multiplication
    if tx < size and ty < size:
        sum = 0.0
        for k in range(size):
            sum += a_shared[tx, k] * b_shared[k, ty]

        # Write the result to the output tensor
        out_idx = tx * size + ty
        out[out_idx] = sum


jit_mm_practice = jit(_mm_practice)


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
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    raise NotImplementedError("Need to implement for Task 3.4")


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
