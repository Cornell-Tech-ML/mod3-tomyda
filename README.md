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

# Task 1.5

```
/usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch  0  loss  8.601916550344345 correct 24
Epoch  10  loss  5.903085305333699 correct 40
Epoch  20  loss  3.9540731195480756 correct 28
Epoch  30  loss  5.958931651707339 correct 38
Epoch  40  loss  4.9083450022229105 correct 45
Epoch  50  loss  3.4842004257185235 correct 48
Epoch  60  loss  4.105810530313646 correct 49
Epoch  70  loss  2.1675764480305526 correct 49
Epoch  80  loss  6.386501474898881 correct 40
Epoch  90  loss  2.1235394516275274 correct 46
Epoch  100  loss  2.7509080787127584 correct 41
Epoch  110  loss  2.4099218100987194 correct 49
Epoch  120  loss  1.9029156360383757 correct 47
Epoch  130  loss  2.6736281973798026 correct 43
Epoch  140  loss  1.4647704991273875 correct 50
Epoch  150  loss  0.9940440249013185 correct 50
Epoch  160  loss  0.8463064010053025 correct 50
Epoch  170  loss  0.9008822897446384 correct 50
Epoch  180  loss  0.4269071144209959 correct 50
Epoch  190  loss  1.3392777064698498 correct 49
Epoch  200  loss  0.44995852962234023 correct 50
Epoch  210  loss  0.6132606501091107 correct 50
Epoch  220  loss  0.6481292184622426 correct 50
Epoch  230  loss  0.4668689917965029 correct 50
Epoch  240  loss  1.0415120735405605 correct 50
Epoch  250  loss  1.1070372755282887 correct 50
Epoch  260  loss  1.0953808801250569 correct 50
Epoch  270  loss  0.3863824621665981 correct 50
Epoch  280  loss  0.53855786908355 correct 49
Epoch  290  loss  1.0789959622877634 correct 50
Epoch  300  loss  0.9202270967286595 correct 50
Epoch  310  loss  1.1822923582168685 correct 49
Epoch  320  loss  0.3166947814292042 correct 50
Epoch  330  loss  1.129788578646751 correct 49
Epoch  340  loss  0.28856512147511193 correct 50
Epoch  350  loss  0.23774834523778185 correct 50
Epoch  360  loss  0.20473281792945816 correct 50
Epoch  370  loss  0.3525654019704084 correct 50
Epoch  380  loss  0.5710736709517492 correct 50
Epoch  390  loss  0.7725176368487535 correct 50
Epoch  400  loss  0.22288026839203723 correct 50
Epoch  410  loss  0.2780132844404524 correct 50
Epoch  420  loss  1.0588894348033184 correct 49
Epoch  430  loss  0.7711781169664963 correct 50
Epoch  440  loss  0.7818209851469873 correct 50
Epoch  450  loss  0.6597907129363779 correct 50$$
Epoch  460  loss  0.1380080270663799 correct 50
Epoch  470  loss  0.14368237061744552 correct 50
Epoch  480  loss  0.02601843434120959 correct 50
Epoch  490  loss  0.10969056399590718 correct 50
```