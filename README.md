AscendC ComplexGram fusion operator
===================================

This project is a reference framework for the requested fused operator:

  Ar/Ai  : [272, 256, 8*n]
  B      : [17, 8*n, 8*n]
  BPlur  : [17, 8*n, 8*n]
  BPlui  : [17, 8*n, 8*n]
  Bsum   : [8*n, 8*n]
  Csum   : [n, n]

Math
----

For K = 8*n and every slice A_s = Ar[s] + j*Ai[s], A_s shape is [256,K].

  P_s = A_s^H @ A_s
      = (Ar_s^T Ar_s + Ai_s^T Ai_s)
        + j(Ar_s^T Ai_s - Ai_s^T Ar_s)

For each group g:

  B[g]     = (1/16) * sum_i (real(P)^2 + imag(P)^2)
  BPlur[g] = (1/16) * sum_i real(P)
  BPlui[g] = (1/16) * sum_i imag(P)

Then:

  Bsum      = (1/17) * sum_g B[g]
  Csum[a,b] = (1/17) * sum_g B[g, a*8, b*8]

Project layout
--------------

  op_kernel/complex_gram_single_kernel.cpp
    Single-kernel AIC/AIV template with SetFlag/WaitFlag wrappers and ping-pong
    ring buffer. This is the version for fine-grained cube/vector synchronization.

  op_kernel/complex_gram_fused.cpp
    Earlier safe multi-kernel template: cube kernel + vector epilogue kernel +
    vector reduce kernel, synchronized by stream order.

  op_kernel/complex_gram_tiling.h
    Host/kernel shared tiling struct and default tiling builder.

  op_host/complex_gram_tiling.cpp
    Official-framework-style host tiling skeleton. It can also be built locally as a
    standalone tiling calculator.

  test_complex_gram_golden.py
    NumPy golden math test.

  test_tiling_plan.py
    Verifies 20 cube / 40 vector task mapping and single-kernel flag/ring-buffer plan.

Single-kernel AIC/AIV synchronization model
-------------------------------------------

The requested single-kernel version is in:

  op_kernel/complex_gram_single_kernel.cpp

It uses the same logical task mapping:

  cube c owns task ids: c, c+20, c+40, ...
  vector v maps to:
      pairCube = v / 2
      lane     = v % 2

Therefore:

  cube 0  -> vector 0, 1
  cube 1  -> vector 2, 3
  ...
  cube 19 -> vector 38, 39

Each cube owns a two-slot ping-pong GM workspace buffer. For cube c, buffer buf:

  workspace base = (c * pingPong + buf) * oneTaskTmpElems

Flag layout per cube and buffer:

  ready0 : cube -> vector lane 0
  ready1 : cube -> vector lane 1
  done0  : vector lane 0 -> cube
  done1  : vector lane 1 -> cube

Flag id formula:

  base  = cubeId * flagSlotsPerBuf * pingPong + buf * flagSlotsPerBuf
  ready = base + lane
  done  = base + 2 + lane

Cube side pseudo flow:

  for task in cubeId, cubeId+20, ...:
      buf = localTaskIndex % 2

      if localTaskIndex >= 2:
          WaitFlag(done0 for this buf)
          WaitFlag(done1 for this buf)

      compute 16 slices * 4 real GEMM into workspace[cubeId][buf]

      SetFlag(ready0 for this buf)
      SetFlag(ready1 for this buf)

  drain last one/two buffers:
      WaitFlag(done0)
      WaitFlag(done1)

Vector side pseudo flow:

  pairCube = vectorId / 2
  lane     = vectorId % 2

  for task in pairCube, pairCube+20, ...:
      buf = localTaskIndex % 2

      WaitFlag(ready lane for this buf)
      consume first half of tile if lane=0, second half if lane=1
      write B/BPlur/BPlui
      SetFlag(done lane for this buf)

This is a real producer-consumer schedule:

  AIC writes tile -> ready flag -> AIV consumes tile -> done flag -> AIC reuses buffer

The wrappers are intentionally isolated in complex_gram_single_kernel.cpp:

  NotifyVecLaneReady
  WaitVecLaneReady
  NotifyCubeLaneDone
  WaitCubeLaneDone

Different CANN versions use different HardEvent names/signatures in official AIC/AIV
fusion samples. Replace those four wrappers with the exact SetFlag/WaitFlag calls from
 your installed CANN version. The schedule, flag-id allocation, buffer reuse rule, and
one-cube-two-vector mapping are already encoded and tested.

Why Bsum/Csum should still be a later phase
------------------------------------------

The single-kernel flag version computes B/BPlur/BPlui in one AIC/AIV-fused kernel.
Bsum/Csum require all B[g,:,:] tiles to be complete. That is a global completion
condition across all vector lanes. Ordinary pairwise SetFlag/WaitFlag is suitable for
producer-consumer tile handoff, but not a safe global barrier across all AIV blocks.

Recommended production flow:

  1. complex_gram_single_kernel
       AIC/AIV fine-grained synchronized cube+vector epilogue.
       Produces B/BPlur/BPlui.

  2. complex_gram_vector_reduce_kernel, from op_kernel/complex_gram_fused.cpp
       Stream-ordered after single kernel.
       Produces Bsum/Csum.

This keeps the expensive cube epilogue fused while keeping the global reduction safe.

Tiling
------

Each task is a group/output-tile:

  task = (g, tile_m, tile_n)

Default:

  BM = 32
  BN = 32

For n=16, K=128:

  tileM = 4
  tileN = 4
  taskNum = 17 * 4 * 4 = 272

Cube side per task:

  for 16 slices:
      rr = Ar^T Ar
      ii = Ai^T Ai
      ri = Ar^T Ai
      ir = Ai^T Ar

Vector side per task:

  lane 0 consumes first half of BM*BN tile
  lane 1 consumes second half of BM*BN tile

Workspace
---------

Multi-kernel full-task workspace:

  workspaceBytes = taskNum * 4 * 16 * BM * BN * sizeof(float)

Single-kernel ping-pong workspace:

  singleKernelWorkspaceBytes = cubeBlockNum * pingPong * 4 * 16 * BM * BN * sizeof(float)

For n=16, BM=BN=32:

  multi-kernel workspace = 71,303,168 bytes
  single-kernel workspace = 10,485,760 bytes

Local commands, no CANN required
--------------------------------

Run golden math test, tiling mapping test, and standalone tiling calculator:

  cd /Users/chenlj/workspace/ascendc_complex_gram
  bash scripts/run_cpu_golden.sh

Expected output includes:

  golden test passed
  tiling plan tests passed
  n=16 K=128 BM=32 BN=32
  tileM=4 tileN=4 tasks=272
  cubeBlockDim=20 vectorBlockDim=40 workspace=71303168 bytes singleKernelWorkspace=10485760 bytes

Ascend/CANN integration commands
--------------------------------

On an Ascend environment:

  cd /Users/chenlj/workspace/ascendc_complex_gram
  export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
  source ${ASCEND_HOME}/set_env.sh
  bash scripts/build_ascend.sh

Then integrate these files into an official AscendC custom-op sample or msOpGen project:

  op_kernel/complex_gram_single_kernel.cpp
  op_kernel/complex_gram_fused.cpp
  op_kernel/complex_gram_tiling.h
  op_host/complex_gram_tiling.cpp

Typical official custom-op flow after integration:

  ./build.sh
  ./build_out/*.run

Then run your ACL/AscendCL test app and compare with test_complex_gram_golden.py.

Important notes
---------------

1. Input type is currently:

     using InT = half;

   If Ar/Ai are true float32 in GM, change this to float and configure the Matmul mode
   supported by your CANN/hardware version, or add a cast/prepack stage.

2. op_kernel/complex_gram_single_kernel.cpp is a template intended for an official
   AscendC AIC/AIV fusion project. The Matmul calls need the Matmul tiling data generated
   by your installed CANN version, and the SetFlag/WaitFlag wrappers need to be adjusted
   to the exact HardEvent names/signatures in your CANN version's official fusion sample.

3. complex_gram_fused_kernel.cpp and complex_gram_kernel.cpp are older reference files.
   The official-style versions are now under op_kernel/.
