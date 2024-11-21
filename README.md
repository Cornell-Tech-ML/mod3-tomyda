# MiniTorch Module 3

# Task 3.1:

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

# Paralelization 3.4

### 3.4 Cuda Matrix Multiply vs Naive approach Performance







# Task 3.5

`run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05`
```
Epoch 0 _ Loss: 5.8500 _ Correct: 28 (Time: 23.4691 sec)
Epoch 10 _ Loss: 1.9308 _ Correct: 47 (Time: 0.1234 sec)
Epoch 20 _ Loss: 1.7277 _ Correct: 49 (Time: 0.1221 sec)
Epoch 30 _ Loss: 1.3330 _ Correct: 50 (Time: 0.2134 sec)
Epoch 40 _ Loss: 1.8658 _ Correct: 49 (Time: 0.1254 sec)
Epoch 50 _ Loss: 0.5698 _ Correct: 49 (Time: 0.1221 sec)
Epoch 60 _ Loss: 1.1295 _ Correct: 50 (Time: 0.1229 sec)
Epoch 70 _ Loss: 0.1713 _ Correct: 49 (Time: 0.1341 sec)
Epoch 80 _ Loss: 0.7378 _ Correct: 50 (Time: 0.1242 sec)
Epoch 90 _ Loss: 0.4354 _ Correct: 50 (Time: 0.1216 sec)
Epoch 100 _ Loss: 0.1977 _ Correct: 50 (Time: 0.1225 sec)
Epoch 110 _ Loss: 0.9593 _ Correct: 50 (Time: 0.1222 sec)
Epoch 120 _ Loss: 0.1947 _ Correct: 50 (Time: 0.2154 sec)
Epoch 130 _ Loss: 0.3034 _ Correct: 50 (Time: 0.1212 sec)
Epoch 140 _ Loss: 0.6914 _ Correct: 50 (Time: 0.1264 sec)
Epoch 150 _ Loss: 0.2724 _ Correct: 50 (Time: 0.1227 sec)
Epoch 160 _ Loss: 0.3648 _ Correct: 50 (Time: 0.1239 sec)
Epoch 170 _ Loss: 0.0769 _ Correct: 50 (Time: 0.1202 sec)
Epoch 180 _ Loss: 0.6605 _ Correct: 50 (Time: 0.1220 sec)
Epoch 190 _ Loss: 0.2305 _ Correct: 50 (Time: 0.1221 sec)
Epoch 200 _ Loss: 0.0400 _ Correct: 50 (Time: 0.1302 sec)
Epoch 210 _ Loss: 0.3827 _ Correct: 50 (Time: 0.2265 sec)
Epoch 220 _ Loss: 0.1929 _ Correct: 50 (Time: 0.1225 sec)
Epoch 230 _ Loss: 0.0857 _ Correct: 50 (Time: 0.1227 sec)
Epoch 240 _ Loss: 0.0129 _ Correct: 50 (Time: 0.1218 sec)
Epoch 250 _ Loss: 0.7145 _ Correct: 50 (Time: 0.1270 sec)
Epoch 260 _ Loss: 0.1532 _ Correct: 50 (Time: 0.1212 sec)
Epoch 270 _ Loss: 0.2067 _ Correct: 50 (Time: 0.1261 sec)
Epoch 280 _ Loss: 0.0051 _ Correct: 50 (Time: 0.1205 sec)
Epoch 290 _ Loss: 0.1212 _ Correct: 50 (Time: 0.1299 sec)
Epoch 300 _ Loss: 0.1371 _ Correct: 50 (Time: 0.2178 sec)
Epoch 310 _ Loss: 0.4203 _ Correct: 50 (Time: 0.1229 sec)
Epoch 320 _ Loss: 0.2873 _ Correct: 50 (Time: 0.1247 sec)
Epoch 330 _ Loss: 0.0061 _ Correct: 50 (Time: 0.1357 sec)
Epoch 340 _ Loss: 0.1110 _ Correct: 50 (Time: 0.1261 sec)
Epoch 350 _ Loss: 0.3389 _ Correct: 50 (Time: 0.1215 sec)
Epoch 360 _ Loss: 0.0132 _ Correct: 50 (Time: 0.1218 sec)
Epoch 370 _ Loss: 0.3720 _ Correct: 50 (Time: 0.1208 sec)
Epoch 380 _ Loss: 0.3743 _ Correct: 50 (Time: 0.1221 sec)
Epoch 390 _ Loss: 0.0016 _ Correct: 50 (Time: 0.2633 sec)
Epoch 400 _ Loss: 0.3948 _ Correct: 50 (Time: 0.1229 sec)
Epoch 410 _ Loss: 0.0008 _ Correct: 50 (Time: 0.1209 sec)
Epoch 420 _ Loss: 0.2279 _ Correct: 50 (Time: 0.1219 sec)
Epoch 430 _ Loss: 0.0209 _ Correct: 50 (Time: 0.1211 sec)
Epoch 440 _ Loss: 0.0263 _ Correct: 50 (Time: 0.1217 sec)
Epoch 450 _ Loss: 0.3518 _ Correct: 50 (Time: 0.1218 sec)
Epoch 460 _ Loss: 0.0009 _ Correct: 50 (Time: 0.1255 sec)
Epoch 470 _ Loss: 0.2110 _ Correct: 50 (Time: 0.1243 sec)
Epoch 480 _ Loss: 0.0801 _ Correct: 50 (Time: 0.2129 sec)
Epoch 490 _ Loss: 0.0918 _ Correct: 50 (Time: 0.1215 sec)

Average Time per Epoch: 0.1852 sec

```

`run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05`
```
Epoch 0 _ Loss: 5.6910 _ Correct: 38 (Time: 5.1595 sec)
Epoch 10 _ Loss: 1.2581 _ Correct: 49 (Time: 1.8028 sec)
Epoch 20 _ Loss: 0.4158 _ Correct: 50 (Time: 1.8748 sec)
Epoch 30 _ Loss: 1.3702 _ Correct: 50 (Time: 1.8057 sec)
Epoch 40 _ Loss: 0.2061 _ Correct: 50 (Time: 1.7978 sec)
Epoch 50 _ Loss: 0.7206 _ Correct: 49 (Time: 2.4567 sec)
Epoch 60 _ Loss: 0.6045 _ Correct: 50 (Time: 1.8504 sec)
Epoch 70 _ Loss: 0.9956 _ Correct: 50 (Time: 1.8068 sec)
Epoch 80 _ Loss: 0.5780 _ Correct: 50 (Time: 1.8117 sec)
Epoch 90 _ Loss: 0.1402 _ Correct: 50 (Time: 1.8145 sec)
Epoch 100 _ Loss: 1.0271 _ Correct: 50 (Time: 2.5245 sec)
Epoch 110 _ Loss: 0.1009 _ Correct: 50 (Time: 1.8573 sec)
Epoch 120 _ Loss: 0.5355 _ Correct: 50 (Time: 1.8195 sec)
Epoch 130 _ Loss: 0.3614 _ Correct: 50 (Time: 1.8629 sec)
Epoch 140 _ Loss: 0.1712 _ Correct: 50 (Time: 1.7726 sec)
Epoch 150 _ Loss: 0.5176 _ Correct: 50 (Time: 2.5405 sec)
Epoch 160 _ Loss: 0.2831 _ Correct: 50 (Time: 1.8089 sec)
Epoch 170 _ Loss: 0.0058 _ Correct: 50 (Time: 1.8543 sec)
Epoch 180 _ Loss: 0.2634 _ Correct: 50 (Time: 1.7887 sec)
Epoch 190 _ Loss: 0.1248 _ Correct: 50 (Time: 1.7771 sec)
Epoch 200 _ Loss: 0.0036 _ Correct: 50 (Time: 2.5828 sec)
Epoch 210 _ Loss: 0.0579 _ Correct: 50 (Time: 1.8492 sec)
Epoch 220 _ Loss: 0.3132 _ Correct: 50 (Time: 1.8015 sec)
Epoch 230 _ Loss: 0.0735 _ Correct: 50 (Time: 1.8005 sec)
Epoch 240 _ Loss: 0.3222 _ Correct: 50 (Time: 1.8819 sec)
Epoch 250 _ Loss: 0.0609 _ Correct: 50 (Time: 2.5756 sec)
Epoch 260 _ Loss: 0.2250 _ Correct: 50 (Time: 1.7904 sec)
Epoch 270 _ Loss: 0.0107 _ Correct: 50 (Time: 1.8363 sec)
Epoch 280 _ Loss: 0.1202 _ Correct: 50 (Time: 1.8486 sec)
Epoch 290 _ Loss: 0.1161 _ Correct: 50 (Time: 1.7971 sec)
Epoch 300 _ Loss: 0.2237 _ Correct: 50 (Time: 2.2907 sec)
Epoch 310 _ Loss: 0.0681 _ Correct: 50 (Time: 1.7939 sec)
Epoch 320 _ Loss: 0.0069 _ Correct: 50 (Time: 1.9595 sec)
Epoch 330 _ Loss: 0.2891 _ Correct: 50 (Time: 1.8248 sec)
Epoch 340 _ Loss: 0.0990 _ Correct: 50 (Time: 1.8077 sec)
Epoch 350 _ Loss: 0.1802 _ Correct: 50 (Time: 2.2916 sec)
Epoch 360 _ Loss: 0.0004 _ Correct: 50 (Time: 1.7929 sec)
Epoch 370 _ Loss: 0.0813 _ Correct: 50 (Time: 2.0216 sec)
Epoch 380 _ Loss: 0.2783 _ Correct: 50 (Time: 1.7766 sec)
Epoch 390 _ Loss: 0.0267 _ Correct: 50 (Time: 1.8589 sec)
Epoch 400 _ Loss: 0.2486 _ Correct: 50 (Time: 1.9805 sec)
Epoch 410 _ Loss: 0.1940 _ Correct: 50 (Time: 1.8023 sec)
Epoch 420 _ Loss: 0.1313 _ Correct: 50 (Time: 2.1971 sec)
Epoch 430 _ Loss: 0.0152 _ Correct: 50 (Time: 1.8360 sec)
Epoch 440 _ Loss: 0.0736 _ Correct: 50 (Time: 1.8303 sec)
Epoch 450 _ Loss: 0.0000 _ Correct: 50 (Time: 1.8851 sec)
Epoch 460 _ Loss: 0.0717 _ Correct: 50 (Time: 1.8463 sec)
Epoch 470 _ Loss: 0.0809 _ Correct: 50 (Time: 2.4089 sec)
Epoch 480 _ Loss: 0.1223 _ Correct: 50 (Time: 1.7982 sec)
Epoch 490 _ Loss: 0.0550 _ Correct: 50 (Time: 1.8033 sec)

Average Time per Epoch: 1.9532 sec
```

`run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
Epoch 0 _ Loss: 11.4109 _ Correct: 19 (Time: 23.1336 sec)
Epoch 10 _ Loss: 5.4070 _ Correct: 42 (Time: 0.1212 sec)
Epoch 20 _ Loss: 5.6538 _ Correct: 38 (Time: 0.1231 sec)
Epoch 30 _ Loss: 5.3825 _ Correct: 45 (Time: 0.2703 sec)
Epoch 40 _ Loss: 4.7004 _ Correct: 45 (Time: 0.1257 sec)
Epoch 50 _ Loss: 5.0797 _ Correct: 41 (Time: 0.1215 sec)
Epoch 60 _ Loss: 3.2323 _ Correct: 48 (Time: 0.1235 sec)
Epoch 70 _ Loss: 3.6041 _ Correct: 45 (Time: 0.1246 sec)
Epoch 80 _ Loss: 4.6023 _ Correct: 45 (Time: 0.1228 sec)
Epoch 90 _ Loss: 3.1354 _ Correct: 44 (Time: 0.1236 sec)
Epoch 100 _ Loss: 3.6613 _ Correct: 47 (Time: 0.1223 sec)
Epoch 110 _ Loss: 2.5628 _ Correct: 48 (Time: 0.1524 sec)
Epoch 120 _ Loss: 1.1894 _ Correct: 46 (Time: 0.2562 sec)
Epoch 130 _ Loss: 3.1551 _ Correct: 49 (Time: 0.1255 sec)
Epoch 140 _ Loss: 1.2585 _ Correct: 47 (Time: 0.1289 sec)
Epoch 150 _ Loss: 2.3770 _ Correct: 49 (Time: 0.1331 sec)
Epoch 160 _ Loss: 2.3838 _ Correct: 47 (Time: 0.1230 sec)
Epoch 170 _ Loss: 1.8901 _ Correct: 48 (Time: 0.1221 sec)
Epoch 180 _ Loss: 1.1607 _ Correct: 49 (Time: 0.1240 sec)
Epoch 190 _ Loss: 1.7135 _ Correct: 47 (Time: 0.1224 sec)
Epoch 200 _ Loss: 0.8507 _ Correct: 48 (Time: 0.2022 sec)
Epoch 210 _ Loss: 1.4003 _ Correct: 48 (Time: 0.2074 sec)
Epoch 220 _ Loss: 0.7066 _ Correct: 47 (Time: 0.1246 sec)
Epoch 230 _ Loss: 3.5320 _ Correct: 46 (Time: 0.1326 sec)
Epoch 240 _ Loss: 0.2858 _ Correct: 49 (Time: 0.1208 sec)
Epoch 250 _ Loss: 0.5056 _ Correct: 42 (Time: 0.1200 sec)
Epoch 260 _ Loss: 0.4904 _ Correct: 47 (Time: 0.1237 sec)
Epoch 270 _ Loss: 0.4157 _ Correct: 49 (Time: 0.1212 sec)
Epoch 280 _ Loss: 0.4958 _ Correct: 49 (Time: 0.1362 sec)
Epoch 290 _ Loss: 0.8683 _ Correct: 48 (Time: 0.2028 sec)
Epoch 300 _ Loss: 2.0807 _ Correct: 48 (Time: 0.1223 sec)
Epoch 310 _ Loss: 1.3621 _ Correct: 48 (Time: 0.1210 sec)
Epoch 320 _ Loss: 1.1233 _ Correct: 50 (Time: 0.1304 sec)
Epoch 330 _ Loss: 2.2510 _ Correct: 45 (Time: 0.1237 sec)
Epoch 340 _ Loss: 0.4289 _ Correct: 48 (Time: 0.1209 sec)
Epoch 350 _ Loss: 1.1468 _ Correct: 47 (Time: 0.1214 sec)
Epoch 360 _ Loss: 0.7534 _ Correct: 48 (Time: 0.1226 sec)
Epoch 370 _ Loss: 1.4222 _ Correct: 48 (Time: 0.1366 sec)
Epoch 380 _ Loss: 2.5551 _ Correct: 45 (Time: 0.1983 sec)
Epoch 390 _ Loss: 1.1109 _ Correct: 48 (Time: 0.1227 sec)
Epoch 400 _ Loss: 0.5336 _ Correct: 48 (Time: 0.1217 sec)
Epoch 410 _ Loss: 1.0171 _ Correct: 48 (Time: 0.1273 sec)
Epoch 420 _ Loss: 0.8815 _ Correct: 48 (Time: 0.1230 sec)
Epoch 430 _ Loss: 3.2484 _ Correct: 49 (Time: 0.1229 sec)
Epoch 440 _ Loss: 1.7669 _ Correct: 48 (Time: 0.1308 sec)
Epoch 450 _ Loss: 2.7237 _ Correct: 48 (Time: 0.1236 sec)
Epoch 460 _ Loss: 1.3187 _ Correct: 48 (Time: 0.1262 sec)
Epoch 470 _ Loss: 1.1683 _ Correct: 50 (Time: 0.1940 sec)
Epoch 480 _ Loss: 0.6786 _ Correct: 48 (Time: 0.1243 sec)
Epoch 490 _ Loss: 0.7250 _ Correct: 48 (Time: 0.1241 sec)

Average Time per Epoch: 0.1845 sec
```

`run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
Epoch 0 _ Loss: 5.9563 _ Correct: 27 (Time: 4.8552 sec)
Epoch 10 _ Loss: 5.1475 _ Correct: 30 (Time: 1.7854 sec)
Epoch 20 _ Loss: 4.1518 _ Correct: 43 (Time: 1.8777 sec)
Epoch 30 _ Loss: 4.1026 _ Correct: 43 (Time: 1.7959 sec)
Epoch 40 _ Loss: 3.1881 _ Correct: 40 (Time: 1.7920 sec)
Epoch 50 _ Loss: 2.4224 _ Correct: 44 (Time: 1.9899 sec)
Epoch 60 _ Loss: 1.9672 _ Correct: 49 (Time: 1.8400 sec)
Epoch 70 _ Loss: 2.3007 _ Correct: 49 (Time: 2.3661 sec)
Epoch 80 _ Loss: 1.1694 _ Correct: 48 (Time: 1.7581 sec)
Epoch 90 _ Loss: 1.1508 _ Correct: 50 (Time: 1.7827 sec)
Epoch 100 _ Loss: 0.1798 _ Correct: 49 (Time: 1.8594 sec)
Epoch 110 _ Loss: 2.2397 _ Correct: 50 (Time: 1.7874 sec)
Epoch 120 _ Loss: 1.0753 _ Correct: 50 (Time: 2.5798 sec)
Epoch 130 _ Loss: 1.2171 _ Correct: 49 (Time: 1.8871 sec)
Epoch 140 _ Loss: 0.5102 _ Correct: 50 (Time: 1.7818 sec)
Epoch 150 _ Loss: 0.6677 _ Correct: 50 (Time: 1.7862 sec)
Epoch 160 _ Loss: 0.8712 _ Correct: 50 (Time: 1.7863 sec)
Epoch 170 _ Loss: 0.5761 _ Correct: 50 (Time: 2.0992 sec)
Epoch 180 _ Loss: 0.6127 _ Correct: 50 (Time: 1.8166 sec)
Epoch 190 _ Loss: 0.7775 _ Correct: 50 (Time: 2.2768 sec)
Epoch 200 _ Loss: 0.1462 _ Correct: 50 (Time: 1.7906 sec)
Epoch 210 _ Loss: 0.5144 _ Correct: 50 (Time: 1.8507 sec)
Epoch 220 _ Loss: 0.2332 _ Correct: 50 (Time: 1.7722 sec)
Epoch 230 _ Loss: 0.0341 _ Correct: 50 (Time: 1.7791 sec)
Epoch 240 _ Loss: 0.9252 _ Correct: 50 (Time: 2.5560 sec)
Epoch 250 _ Loss: 0.7171 _ Correct: 50 (Time: 1.8245 sec)
Epoch 260 _ Loss: 1.0734 _ Correct: 50 (Time: 1.8359 sec)
Epoch 270 _ Loss: 0.2293 _ Correct: 50 (Time: 1.7644 sec)
Epoch 280 _ Loss: 0.7509 _ Correct: 50 (Time: 1.8525 sec)
Epoch 290 _ Loss: 0.0329 _ Correct: 50 (Time: 2.0354 sec)
Epoch 300 _ Loss: 0.2635 _ Correct: 50 (Time: 1.7918 sec)
Epoch 310 _ Loss: 1.2648 _ Correct: 48 (Time: 2.2552 sec)
Epoch 320 _ Loss: 0.8698 _ Correct: 50 (Time: 1.8525 sec)
Epoch 330 _ Loss: 0.2168 _ Correct: 50 (Time: 1.7949 sec)
Epoch 340 _ Loss: 0.4986 _ Correct: 50 (Time: 1.7892 sec)
Epoch 350 _ Loss: 0.5005 _ Correct: 50 (Time: 1.8680 sec)
Epoch 360 _ Loss: 0.5281 _ Correct: 50 (Time: 2.5937 sec)
Epoch 370 _ Loss: 0.6188 _ Correct: 50 (Time: 1.8219 sec)
Epoch 380 _ Loss: 0.0587 _ Correct: 50 (Time: 1.8040 sec)
Epoch 390 _ Loss: 0.1272 _ Correct: 50 (Time: 1.8521 sec)
Epoch 400 _ Loss: 0.2870 _ Correct: 50 (Time: 1.8318 sec)
Epoch 410 _ Loss: 0.2136 _ Correct: 50 (Time: 2.3066 sec)
Epoch 420 _ Loss: 0.2289 _ Correct: 50 (Time: 1.8022 sec)
Epoch 430 _ Loss: 0.0315 _ Correct: 50 (Time: 2.0879 sec)
Epoch 440 _ Loss: 0.2657 _ Correct: 50 (Time: 1.7832 sec)
Epoch 450 _ Loss: 0.0216 _ Correct: 50 (Time: 1.7848 sec)
Epoch 460 _ Loss: 0.2675 _ Correct: 50 (Time: 1.9772 sec)
Epoch 470 _ Loss: 0.3357 _ Correct: 50 (Time: 1.7843 sec)
Epoch 480 _ Loss: 0.2725 _ Correct: 50 (Time: 2.3730 sec)
Epoch 490 _ Loss: 0.3085 _ Correct: 50 (Time: 1.7807 sec)

Average Time per Epoch: 1.9900 sec
```


`run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
Epoch 0 _ Loss: 11.6573 _ Correct: 22 (Time: 23.5368 sec)
Epoch 10 _ Loss: 4.7629 _ Correct: 34 (Time: 0.1270 sec)
Epoch 20 _ Loss: 4.4080 _ Correct: 44 (Time: 0.1293 sec)
Epoch 30 _ Loss: 3.3322 _ Correct: 45 (Time: 0.1247 sec)
Epoch 40 _ Loss: 2.5811 _ Correct: 45 (Time: 0.1245 sec)
Epoch 50 _ Loss: 1.6502 _ Correct: 48 (Time: 0.1254 sec)
Epoch 60 _ Loss: 2.3234 _ Correct: 49 (Time: 0.1248 sec)
Epoch 70 _ Loss: 4.0870 _ Correct: 47 (Time: 0.2057 sec)
Epoch 80 _ Loss: 1.2477 _ Correct: 50 (Time: 0.1247 sec)
Epoch 90 _ Loss: 1.3015 _ Correct: 50 (Time: 0.1393 sec)
Epoch 100 _ Loss: 1.4661 _ Correct: 48 (Time: 0.1282 sec)
Epoch 110 _ Loss: 1.7446 _ Correct: 50 (Time: 0.1264 sec)
Epoch 120 _ Loss: 1.2499 _ Correct: 50 (Time: 0.1248 sec)
Epoch 130 _ Loss: 1.2604 _ Correct: 49 (Time: 0.1351 sec)
Epoch 140 _ Loss: 0.9094 _ Correct: 50 (Time: 0.1275 sec)
Epoch 150 _ Loss: 0.8433 _ Correct: 50 (Time: 0.1909 sec)
Epoch 160 _ Loss: 0.1969 _ Correct: 50 (Time: 0.1237 sec)
Epoch 170 _ Loss: 0.7805 _ Correct: 50 (Time: 0.1307 sec)
Epoch 180 _ Loss: 1.0369 _ Correct: 50 (Time: 0.1247 sec)
Epoch 190 _ Loss: 0.9958 _ Correct: 50 (Time: 0.1266 sec)
Epoch 200 _ Loss: 0.9437 _ Correct: 50 (Time: 0.1255 sec)
Epoch 210 _ Loss: 1.1024 _ Correct: 50 (Time: 0.1240 sec)
Epoch 220 _ Loss: 0.6669 _ Correct: 50 (Time: 0.1249 sec)
Epoch 230 _ Loss: 0.3544 _ Correct: 50 (Time: 0.1254 sec)
Epoch 240 _ Loss: 0.9801 _ Correct: 50 (Time: 0.2631 sec)
Epoch 250 _ Loss: 0.6828 _ Correct: 50 (Time: 0.1242 sec)
Epoch 260 _ Loss: 0.6725 _ Correct: 50 (Time: 0.1257 sec)
Epoch 270 _ Loss: 0.3931 _ Correct: 50 (Time: 0.1284 sec)
Epoch 280 _ Loss: 0.6142 _ Correct: 50 (Time: 0.1239 sec)
Epoch 290 _ Loss: 0.2309 _ Correct: 50 (Time: 0.1258 sec)
Epoch 300 _ Loss: 0.5085 _ Correct: 50 (Time: 0.1246 sec)
Epoch 310 _ Loss: 0.3470 _ Correct: 50 (Time: 0.1261 sec)
Epoch 320 _ Loss: 1.0337 _ Correct: 50 (Time: 0.1241 sec)
Epoch 330 _ Loss: 0.1836 _ Correct: 50 (Time: 0.2618 sec)
Epoch 340 _ Loss: 0.3876 _ Correct: 50 (Time: 0.1345 sec)
Epoch 350 _ Loss: 0.0701 _ Correct: 50 (Time: 0.1228 sec)
Epoch 360 _ Loss: 0.3182 _ Correct: 50 (Time: 0.1360 sec)
Epoch 370 _ Loss: 0.2601 _ Correct: 50 (Time: 0.1253 sec)
Epoch 380 _ Loss: 0.8374 _ Correct: 50 (Time: 0.1246 sec)
Epoch 390 _ Loss: 0.1676 _ Correct: 50 (Time: 0.1267 sec)
Epoch 400 _ Loss: 0.7548 _ Correct: 50 (Time: 0.1315 sec)
Epoch 410 _ Loss: 0.1595 _ Correct: 50 (Time: 0.1300 sec)
Epoch 420 _ Loss: 0.2897 _ Correct: 50 (Time: 0.2732 sec)
Epoch 430 _ Loss: 0.8232 _ Correct: 50 (Time: 0.1250 sec)
Epoch 440 _ Loss: 0.1889 _ Correct: 50 (Time: 0.1372 sec)
Epoch 450 _ Loss: 0.1418 _ Correct: 50 (Time: 0.1251 sec)
Epoch 460 _ Loss: 0.1051 _ Correct: 50 (Time: 0.1256 sec)
Epoch 470 _ Loss: 0.0711 _ Correct: 50 (Time: 0.1283 sec)
Epoch 480 _ Loss: 0.1980 _ Correct: 50 (Time: 0.1251 sec)
Epoch 490 _ Loss: 0.3221 _ Correct: 50 (Time: 0.1240 sec)

Average Time per Epoch: 0.6065 sec
```

`run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
Epoch 0 _ Loss: 8.3610 _ Correct: 27 (Time: 4.5124 sec)
Epoch 10 _ Loss: 4.0932 _ Correct: 48 (Time: 2.3596 sec)
Epoch 20 _ Loss: 5.4195 _ Correct: 39 (Time: 1.8823 sec)
Epoch 30 _ Loss: 3.0951 _ Correct: 48 (Time: 1.7826 sec)
Epoch 40 _ Loss: 2.0142 _ Correct: 49 (Time: 1.8237 sec)
Epoch 50 _ Loss: 1.8085 _ Correct: 50 (Time: 1.7927 sec)
Epoch 60 _ Loss: 2.2626 _ Correct: 49 (Time: 2.5006 sec)
Epoch 70 _ Loss: 3.5491 _ Correct: 47 (Time: 1.8267 sec)
Epoch 80 _ Loss: 1.6692 _ Correct: 50 (Time: 1.8096 sec)
Epoch 90 _ Loss: 1.1677 _ Correct: 49 (Time: 1.8211 sec)
Epoch 100 _ Loss: 1.7205 _ Correct: 49 (Time: 1.8550 sec)
Epoch 110 _ Loss: 1.5221 _ Correct: 50 (Time: 2.5549 sec)
Epoch 120 _ Loss: 1.0159 _ Correct: 49 (Time: 1.8084 sec)
Epoch 130 _ Loss: 1.2269 _ Correct: 50 (Time: 1.8667 sec)
Epoch 140 _ Loss: 0.1738 _ Correct: 49 (Time: 1.7793 sec)
Epoch 150 _ Loss: 1.5075 _ Correct: 50 (Time: 1.7749 sec)
Epoch 160 _ Loss: 0.3907 _ Correct: 50 (Time: 2.0111 sec)
Epoch 170 _ Loss: 0.3252 _ Correct: 50 (Time: 1.8580 sec)
Epoch 180 _ Loss: 1.0727 _ Correct: 49 (Time: 2.3616 sec)
Epoch 190 _ Loss: 0.1170 _ Correct: 50 (Time: 1.7849 sec)
Epoch 200 _ Loss: 1.2060 _ Correct: 49 (Time: 1.7604 sec)
Epoch 210 _ Loss: 0.4209 _ Correct: 50 (Time: 1.8342 sec)
Epoch 220 _ Loss: 0.7785 _ Correct: 49 (Time: 1.7728 sec)
Epoch 230 _ Loss: 0.3336 _ Correct: 50 (Time: 2.3827 sec)
Epoch 240 _ Loss: 0.2704 _ Correct: 50 (Time: 1.8613 sec)
Epoch 250 _ Loss: 1.0212 _ Correct: 49 (Time: 1.8216 sec)
Epoch 260 _ Loss: 0.2732 _ Correct: 50 (Time: 1.7930 sec)
Epoch 270 _ Loss: 0.9478 _ Correct: 50 (Time: 1.8002 sec)
Epoch 280 _ Loss: 0.1756 _ Correct: 50 (Time: 2.4769 sec)
Epoch 290 _ Loss: 0.2376 _ Correct: 50 (Time: 1.7932 sec)
Epoch 300 _ Loss: 0.2103 _ Correct: 50 (Time: 1.8005 sec)
Epoch 310 _ Loss: 0.7612 _ Correct: 50 (Time: 1.7799 sec)
Epoch 320 _ Loss: 0.8433 _ Correct: 50 (Time: 1.8520 sec)
Epoch 330 _ Loss: 0.7491 _ Correct: 50 (Time: 2.4186 sec)
Epoch 340 _ Loss: 0.2271 _ Correct: 50 (Time: 1.7838 sec)
Epoch 350 _ Loss: 0.2694 _ Correct: 50 (Time: 1.9374 sec)
Epoch 360 _ Loss: 0.1461 _ Correct: 50 (Time: 1.7902 sec)
Epoch 370 _ Loss: 0.1717 _ Correct: 50 (Time: 1.7932 sec)
Epoch 380 _ Loss: 0.1887 _ Correct: 50 (Time: 1.9940 sec)
Epoch 390 _ Loss: 0.1233 _ Correct: 50 (Time: 1.8617 sec)
Epoch 400 _ Loss: 0.1409 _ Correct: 50 (Time: 2.3187 sec)
Epoch 410 _ Loss: 0.0979 _ Correct: 50 (Time: 1.8489 sec)
Epoch 420 _ Loss: 0.1169 _ Correct: 50 (Time: 1.7749 sec)
Epoch 430 _ Loss: 0.7031 _ Correct: 50 (Time: 1.8547 sec)
Epoch 440 _ Loss: 0.1604 _ Correct: 50 (Time: 1.7799 sec)
Epoch 450 _ Loss: 0.6680 _ Correct: 50 (Time: 2.5825 sec)
Epoch 460 _ Loss: 0.0679 _ Correct: 50 (Time: 1.8608 sec)
Epoch 470 _ Loss: 0.6814 _ Correct: 50 (Time: 1.7709 sec)
Epoch 480 _ Loss: 0.6806 _ Correct: 50 (Time: 1.7874 sec)
Epoch 490 _ Loss: 0.0571 _ Correct: 50 (Time: 1.7699 sec)

Average Time per Epoch: 1.9429 sec
```

### Bigger models

`run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET simple --RATE 0.05`
```
Epoch 0 _ Loss: 15.2001 _ Correct: 23 (Time: 23.3941 sec)
Epoch 10 _ Loss: 3.2220 _ Correct: 45 (Time: 0.2794 sec)
Epoch 20 _ Loss: 1.4389 _ Correct: 50 (Time: 0.2753 sec)
Epoch 30 _ Loss: 1.3006 _ Correct: 50 (Time: 0.2656 sec)
Epoch 40 _ Loss: 0.4353 _ Correct: 49 (Time: 0.2738 sec)
Epoch 50 _ Loss: 0.3301 _ Correct: 49 (Time: 0.2640 sec)
Epoch 60 _ Loss: 0.3177 _ Correct: 49 (Time: 0.2629 sec)
Epoch 70 _ Loss: 0.2693 _ Correct: 49 (Time: 0.2771 sec)
Epoch 80 _ Loss: 0.3205 _ Correct: 50 (Time: 0.2603 sec)
Epoch 90 _ Loss: 0.9594 _ Correct: 50 (Time: 0.2608 sec)
Epoch 100 _ Loss: 0.0897 _ Correct: 49 (Time: 0.2760 sec)
Epoch 110 _ Loss: 0.0555 _ Correct: 49 (Time: 0.2626 sec)
Epoch 120 _ Loss: 0.1823 _ Correct: 50 (Time: 0.2631 sec)
Epoch 130 _ Loss: 0.2281 _ Correct: 50 (Time: 0.2762 sec)
Epoch 140 _ Loss: 0.1193 _ Correct: 50 (Time: 0.3040 sec)
Epoch 150 _ Loss: 0.0631 _ Correct: 50 (Time: 0.2640 sec)
Epoch 160 _ Loss: 0.1136 _ Correct: 50 (Time: 0.2618 sec)
Epoch 170 _ Loss: 0.6176 _ Correct: 49 (Time: 0.2634 sec)
Epoch 180 _ Loss: 0.0497 _ Correct: 50 (Time: 0.4739 sec)
Epoch 190 _ Loss: 0.0491 _ Correct: 50 (Time: 0.2641 sec)
Epoch 200 _ Loss: 0.8072 _ Correct: 50 (Time: 0.2618 sec)
Epoch 210 _ Loss: 0.0348 _ Correct: 50 (Time: 0.2752 sec)
Epoch 220 _ Loss: 0.6164 _ Correct: 50 (Time: 0.5371 sec)
Epoch 230 _ Loss: 0.1037 _ Correct: 50 (Time: 0.2652 sec)
Epoch 240 _ Loss: 0.0026 _ Correct: 50 (Time: 0.2775 sec)
Epoch 250 _ Loss: 0.7380 _ Correct: 50 (Time: 0.2643 sec)
Epoch 260 _ Loss: 0.1135 _ Correct: 50 (Time: 0.4176 sec)
Epoch 270 _ Loss: 0.0314 _ Correct: 50 (Time: 0.2729 sec)
Epoch 280 _ Loss: 0.4865 _ Correct: 50 (Time: 0.2798 sec)
Epoch 290 _ Loss: 0.5445 _ Correct: 50 (Time: 0.2632 sec)
Epoch 300 _ Loss: 0.5397 _ Correct: 50 (Time: 0.2756 sec)
Epoch 310 _ Loss: 0.0089 _ Correct: 50 (Time: 0.2616 sec)
Epoch 320 _ Loss: 0.0142 _ Correct: 50 (Time: 0.2638 sec)
Epoch 330 _ Loss: 0.1268 _ Correct: 50 (Time: 0.2616 sec)
Epoch 340 _ Loss: 0.0008 _ Correct: 50 (Time: 0.2628 sec)
Epoch 350 _ Loss: 0.0486 _ Correct: 50 (Time: 0.2605 sec)
Epoch 360 _ Loss: 0.4749 _ Correct: 50 (Time: 0.2586 sec)
Epoch 370 _ Loss: 0.0150 _ Correct: 50 (Time: 0.2648 sec)
Epoch 380 _ Loss: 0.0419 _ Correct: 50 (Time: 0.2723 sec)
Epoch 390 _ Loss: 0.0161 _ Correct: 50 (Time: 0.2637 sec)
Epoch 400 _ Loss: 0.0332 _ Correct: 50 (Time: 0.2672 sec)
Epoch 410 _ Loss: 0.0004 _ Correct: 50 (Time: 0.2733 sec)
Epoch 420 _ Loss: 0.0923 _ Correct: 50 (Time: 0.2678 sec)
Epoch 430 _ Loss: 0.3048 _ Correct: 50 (Time: 0.2689 sec)
Epoch 440 _ Loss: 0.0062 _ Correct: 50 (Time: 0.2753 sec)
Epoch 450 _ Loss: 0.0295 _ Correct: 50 (Time: 0.2648 sec)
Epoch 460 _ Loss: 0.0454 _ Correct: 50 (Time: 0.2627 sec)
Epoch 470 _ Loss: 0.2925 _ Correct: 50 (Time: 0.4910 sec)
Epoch 480 _ Loss: 0.3172 _ Correct: 50 (Time: 0.2641 sec)
Epoch 490 _ Loss: 0.2529 _ Correct: 50 (Time: 0.2633 sec)

Average Time per Epoch: 0.3418 sec
```

`run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET simple --RATE 0.05`
```
Epoch 0 _ Loss: 2.9315 _ Correct: 47 (Time: 4.7677 sec)
Epoch 10 _ Loss: 0.2346 _ Correct: 49 (Time: 1.8495 sec)
Epoch 20 _ Loss: 1.2511 _ Correct: 50 (Time: 1.9103 sec)
Epoch 30 _ Loss: 0.6560 _ Correct: 49 (Time: 1.8833 sec)
Epoch 40 _ Loss: 1.4740 _ Correct: 50 (Time: 1.8453 sec)
Epoch 50 _ Loss: 0.3367 _ Correct: 50 (Time: 1.8459 sec)
Epoch 60 _ Loss: 0.6466 _ Correct: 50 (Time: 1.9021 sec)
Epoch 70 _ Loss: 0.1594 _ Correct: 50 (Time: 1.8316 sec)
Epoch 80 _ Loss: 0.0117 _ Correct: 50 (Time: 2.1299 sec)
Epoch 90 _ Loss: 0.0865 _ Correct: 50 (Time: 1.8333 sec)
Epoch 100 _ Loss: 0.1242 _ Correct: 50 (Time: 1.9203 sec)
Epoch 110 _ Loss: 0.3312 _ Correct: 50 (Time: 2.6592 sec)
Epoch 120 _ Loss: 0.0482 _ Correct: 50 (Time: 1.8544 sec)
Epoch 130 _ Loss: 0.0503 _ Correct: 50 (Time: 1.9247 sec)
Epoch 140 _ Loss: 0.1637 _ Correct: 50 (Time: 2.4004 sec)
Epoch 150 _ Loss: 0.2339 _ Correct: 50 (Time: 1.8332 sec)
Epoch 160 _ Loss: 0.1030 _ Correct: 50 (Time: 1.8396 sec)
Epoch 170 _ Loss: 0.3201 _ Correct: 50 (Time: 2.0150 sec)
Epoch 180 _ Loss: 0.0993 _ Correct: 50 (Time: 1.8480 sec)
Epoch 190 _ Loss: 0.1461 _ Correct: 50 (Time: 1.8413 sec)
Epoch 200 _ Loss: 0.5272 _ Correct: 50 (Time: 1.8259 sec)
Epoch 210 _ Loss: 0.0063 _ Correct: 50 (Time: 1.9043 sec)
Epoch 220 _ Loss: 0.0495 _ Correct: 50 (Time: 1.8595 sec)
Epoch 230 _ Loss: 0.3379 _ Correct: 50 (Time: 1.8495 sec)
Epoch 240 _ Loss: 0.1628 _ Correct: 50 (Time: 1.9113 sec)
Epoch 250 _ Loss: 0.0448 _ Correct: 50 (Time: 2.3701 sec)
Epoch 260 _ Loss: 0.0023 _ Correct: 50 (Time: 1.8421 sec)
Epoch 270 _ Loss: 0.1337 _ Correct: 50 (Time: 1.8711 sec)
Epoch 280 _ Loss: 0.0855 _ Correct: 50 (Time: 2.7150 sec)
Epoch 290 _ Loss: 0.1232 _ Correct: 50 (Time: 1.8617 sec)
Epoch 300 _ Loss: 0.0151 _ Correct: 50 (Time: 1.8299 sec)
Epoch 310 _ Loss: 0.0291 _ Correct: 50 (Time: 2.1737 sec)
Epoch 320 _ Loss: 0.0015 _ Correct: 50 (Time: 1.9020 sec)
Epoch 330 _ Loss: 0.1476 _ Correct: 50 (Time: 1.8848 sec)
Epoch 340 _ Loss: 0.0087 _ Correct: 50 (Time: 1.8351 sec)
Epoch 350 _ Loss: 0.0098 _ Correct: 50 (Time: 1.9013 sec)
Epoch 360 _ Loss: 0.3436 _ Correct: 50 (Time: 2.3147 sec)
Epoch 370 _ Loss: 0.0422 _ Correct: 50 (Time: 1.8363 sec)
Epoch 380 _ Loss: 0.1615 _ Correct: 50 (Time: 1.8341 sec)
Epoch 390 _ Loss: 0.1385 _ Correct: 50 (Time: 2.5249 sec)
Epoch 400 _ Loss: 0.0842 _ Correct: 50 (Time: 1.8231 sec)
Epoch 410 _ Loss: 0.1000 _ Correct: 50 (Time: 1.8426 sec)
Epoch 420 _ Loss: 0.0547 _ Correct: 50 (Time: 2.3778 sec)
Epoch 430 _ Loss: 0.0192 _ Correct: 50 (Time: 1.9079 sec)
Epoch 440 _ Loss: 0.0333 _ Correct: 50 (Time: 1.8333 sec)
Epoch 450 _ Loss: 0.1138 _ Correct: 50 (Time: 1.8274 sec)
Epoch 460 _ Loss: 0.0006 _ Correct: 50 (Time: 1.9050 sec)
Epoch 470 _ Loss: 0.0001 _ Correct: 50 (Time: 2.1726 sec)
Epoch 480 _ Loss: 0.1897 _ Correct: 50 (Time: 1.8526 sec)
Epoch 490 _ Loss: 0.0224 _ Correct: 50 (Time: 1.8392 sec)

Average Time per Epoch: 2.0044 sec
```
