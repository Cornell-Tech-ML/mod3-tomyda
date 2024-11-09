# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# Task 3.1 & 3.2:

```
(.venv) ➜  mod3-tomyda git:(master) python project/parallel_check.py

MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/tommy/Code/cornell code/mod3-tomyda/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/tommy/Code/cornell code/mod3-tomyda/minitorch/fast_ops.py (164)
------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                             |
        out: Storage,                                                                     |
        out_shape: Shape,                                                                 |
        out_strides: Strides,                                                             |
        in_storage: Storage,                                                              |
        in_shape: Shape,                                                                  |
        in_strides: Strides,                                                              |
    ) -> None:                                                                            |
        same_shape = len(out_shape) == len(in_shape) and np.all(out_shape == in_shape)----| #0
        same_strides = len(out_strides) == len(in_strides) and np.all(                    |
            out_strides == in_strides-----------------------------------------------------| #1
        )                                                                                 |
        if same_shape and same_strides:                                                   |
            for i in prange(len(out)):----------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                |
        else:                                                                             |
            for i in prange(len(out)):----------------------------------------------------| #3
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                            |
                in_index = np.empty(MAX_DIMS, dtype=np.int32)                             |
                to_index(i, out_shape, out_index)                                         |
                broadcast_index(out_index, out_shape, in_shape, in_index)                 |
                o = index_to_position(out_index, out_strides)                             |
                j = index_to_position(in_index, in_strides)                               |
                out[o] = fn(in_storage[j])                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/tommy/Code/cornell
code/mod3-tomyda/minitorch/fast_ops.py (181) is hoisted out of the parallel loop
 labelled #3 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/tommy/Code/cornell
code/mod3-tomyda/minitorch/fast_ops.py (182) is hoisted out of the parallel loop
 labelled #3 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/tommy/Code/cornell code/mod3-tomyda/minitorch/fast_ops.py (215)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/tommy/Code/cornell code/mod3-tomyda/minitorch/fast_ops.py (215)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        same_shape = (                                                     |
            len(out_shape) == len(a_shape) == len(b_shape)                 |
            and np.all(out_shape == a_shape)-------------------------------| #4
            and np.all(out_shape == b_shape)-------------------------------| #5
        )                                                                  |
        same_strides = (                                                   |
            len(out_strides) == len(a_strides) == len(b_strides)           |
            and np.all(out_strides == a_strides)---------------------------| #6
            and np.all(out_strides == b_strides)---------------------------| #7
        )                                                                  |
        if same_shape and same_strides:                                    |
            # Stride-aligned, process directly                             |
            for i in prange(len(out)):-------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                    |
        else:                                                              |
            for i in prange(len(out)):-------------------------------------| #9
                out_index = np.empty(MAX_DIMS, dtype=np.int32)             |
                a_index = np.empty(MAX_DIMS, dtype=np.int32)               |
                b_index = np.empty(MAX_DIMS, dtype=np.int32)               |
                to_index(i, out_shape, out_index)                          |
                o = index_to_position(out_index, out_strides)              |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                j = index_to_position(a_index, a_strides)                  |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                k = index_to_position(b_index, b_strides)                  |
                out[o] = fn(a_storage[j], b_storage[k])                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/tommy/Code/cornell
code/mod3-tomyda/minitorch/fast_ops.py (242) is hoisted out of the parallel loop
 labelled #9 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/tommy/Code/cornell
code/mod3-tomyda/minitorch/fast_ops.py (243) is hoisted out of the parallel loop
 labelled #9 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/tommy/Code/cornell
code/mod3-tomyda/minitorch/fast_ops.py (244) is hoisted out of the parallel loop
 labelled #9 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/tommy/Code/cornell code/mod3-tomyda/minitorch/fast_ops.py (277)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/tommy/Code/cornell code/mod3-tomyda/minitorch/fast_ops.py (277)
---------------------------------------------------------------|loop #ID
    def _reduce(                                               |
        out: Storage,                                          |
        out_shape: Shape,                                      |
        out_strides: Strides,                                  |
        a_storage: Storage,                                    |
        a_shape: Shape,                                        |
        a_strides: Strides,                                    |
        reduce_dim: int,                                       |
    ) -> None:                                                 |
        for i in prange(len(out)):-----------------------------| #10
            out_index: Index = np.empty(MAX_DIMS, np.int32)    |
            reduce_size = a_shape[reduce_dim]                  |
            to_index(i, out_shape, out_index)                  |
            o = index_to_position(out_index, out_strides)      |
            accum = out[o]                                     |
            j = index_to_position(out_index, a_strides)        |
            step = a_strides[reduce_dim]                       |
            for s in range(reduce_size):                       |
                accum = fn(accum, a_storage[j])                |
                j += step                                      |
            out[o] = accum                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/tommy/Code/cornell
code/mod3-tomyda/minitorch/fast_ops.py (287) is hoisted out of the parallel loop
 labelled #10 (it will be performed before the loop is executed and reused
inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/tommy/Code/cornell code/mod3-tomyda/minitorch/fast_ops.py (302)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/tommy/Code/cornell code/mod3-tomyda/minitorch/fast_ops.py (302)
---------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                           |
    out: Storage,                                                                      |
    out_shape: Shape,                                                                  |
    out_strides: Strides,                                                              |
    a_storage: Storage,                                                                |
    a_shape: Shape,                                                                    |
    a_strides: Strides,                                                                |
    b_storage: Storage,                                                                |
    b_shape: Shape,                                                                    |
    b_strides: Strides,                                                                |
) -> None:                                                                             |
    """NUMBA tensor matrix multiply function.                                          |
                                                                                       |
    Should work for any tensor shapes that broadcast as long as                        |
                                                                                       |
    ```                                                                                |
    assert a_shape[-1] == b_shape[-2]                                                  |
    ```                                                                                |
                                                                                       |
    Optimizations:                                                                     |
                                                                                       |
    * Outer loop in parallel                                                           |
    * No index buffers or function calls                                               |
    * Inner loop should have no global writes, 1 multiply.                             |
                                                                                       |
                                                                                       |
    Args:                                                                              |
    ----                                                                               |
        out (Storage): storage for `out` tensor                                        |
        out_shape (Shape): shape for `out` tensor                                      |
        out_strides (Strides): strides for `out` tensor                                |
        a_storage (Storage): storage for `a` tensor                                    |
        a_shape (Shape): shape for `a` tensor                                          |
        a_strides (Strides): strides for `a` tensor                                    |
        b_storage (Storage): storage for `b` tensor                                    |
        b_shape (Shape): shape for `b` tensor                                          |
        b_strides (Strides): strides for `b` tensor                                    |
                                                                                       |
    Returns:                                                                           |
    -------                                                                            |
        None : Fills in `out`                                                          |
                                                                                       |
    """                                                                                |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                             |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                             |
                                                                                       |
    for i1 in prange(out_shape[0]):----------------------------------------------------| #13
        for i2 in prange(out_shape[1]):------------------------------------------------| #12
            for i3 in prange(out_shape[2]):--------------------------------------------| #11
                a_inner = i1 * a_batch_stride + i2 * a_strides[1]                      |
                b_inner = i1 * b_batch_stride + i3 * b_strides[2]                      |
                acc = 0.0                                                              |
                for _ in range(a_shape[2]):                                            |
                    acc += a_storage[a_inner] * b_storage[b_inner]                     |
                    a_inner += a_strides[2]                                            |
                    b_inner += b_strides[1]                                            |
                out_position = (                                                       |
                    i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]    |
                )                                                                      |
                out[out_position] = acc                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)



Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
(.venv) ➜  mod3-tomyda git:(master)
```