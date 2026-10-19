## Claims

- **C1. Correct.** Results match a CPU reference across all supported configurations
    - Exactly for u32 ops and f32 min/max, and within a stated tolerance for f32 sum.
    - Inputs are assumed NaN-free, and integer sums are assumed not to overflow.
- **C2. Approachable.** A JS developer can use WAKE with code comparable to plain JS and no GPU-specific knowledge, in contrast to raw WebGPU.
- **C3. Useful.** Covers common operations JS developers need (aggregation and sorting) on the common numeric types, and its primitives compose (scan is reused inside sort).
- **C4. Performant.**
    - (a) Faster than CPU JavaScript in both kernel-only and round-trip time above a characterized input-size threshold.
    - (b) Reduce and scan kernels (which are memory-bound) achieve a high fraction of measured memory bandwidth.
    - (c) Both hold on an integrated and a discrete GPU.

## Scope

**Reduce: sum/min/max over u32/f32**

- *C3:* sum, min, and max are the most common aggregations. f32 is essential because floats are what JS developers mostly work with (`Float32Array`, numeric data); u32 covers integer data.
- *C2:* a single call with an op and dtype option, so users never write a kernel per type.
- *C4:* reduce is memory-bound, so it's a clean test of bandwidth efficiency (C4b). f32 sum is the headline benchmark.

**Scan: exclusive u32 sum only**

- *C3 (composition):* scan's role is as a building block. WAKE's own sort is built on it, which shows the primitives compose. Exclusive u32 sum is exactly the form sort, compaction, and offset computation use.
- *C4:* a second memory-bound primitive, and a harder one, since it needs cross-workgroup coordination. Benchmarked on u32.

**Sort: u32/f32 keys-only, ascending**

- *C3*: sorting is among the most common operations on any data. u32 and f32 keys cover integer and floating-point data. The sort is stable, so key-value sorting can be added without changing the algorithm.
- *C2:* f32 keys are handled internally with a bit transform, so users sort floats as easily as u32.
- *C3/C4*: demonstrates composition (count → scan → reorder). Performance is compared against JavaScript's built-in TypedArray.prototype.sort, not memory bandwidth.

**Assumptions: subgroups available, subgroup size 32**

- *C4:* subgroups are WebGPU’s version of CUDA warp-level primitives, which GPU literature has long used to improve performance. WAKE uses them for the same reason.
- *C2:* WAKE checks these requirements at initialization and fails with a clear error, rather than producing silent wrong results.
- This is a deliberate trade-off: it favors C4 over broader device reach (see out of scope).

**Devices: M2 MacBook Air and NVIDIA RTX 3080**

- *C4c:* spans an integrated GPU with unified memory (Metal) and a discrete GPU over PCIe (D3D12/Vulkan, depending on OS). Two vendors and two native backends show performance isn't tuned to one device.
- *C4a:* the memory architectures differ in transfer cost, so comparing them shows how the CPU/GPU crossover depends on hardware.
- *C1:* correctness on two architectures is stronger evidence than on one.

### Out of scope

**Not needed by any claim:**

- **i32:** f32 covers most signed use cases, and i32 is a mechanical extension of u32.
- **Inclusive scan, scan min/max, f32/i32 scan:** scan's justification is composition (C3), and the form WAKE uses is exclusive u32 sum. The others are straightforward extensions, not new evidence.
- **Arg-reduce (argmin/argmax):** useful, but it adds implementation complexity (pair reductions, tie-breaking) without new evidence for any claim; its memory traffic matches plain reduce.
- **Descending sort:** trivially recovered by reversing, so it adds no evidence for C3.
- **Additional primitives** (matrix multiplication, segmented operations): C3 is supported by covering the most foundational operations and showing composition, not by breadth. These are future work.

**Deliberate trade-offs:**

- **Devices without subgroups, or with a subgroup size other than 32** (some AMD, Intel, and mobile GPUs): excluded in favor of C4 and a tractable implementation. This limits reach, but both evaluated devices meet the requirements. Supporting different subgroup sizes or no subgroups is future work.

**Limits of the evidence:**

- **Two devices:** enough to show performance isn't tuned to one GPU (C4c), but not proof of general portability. Both use a subgroup size of 32.
- **Approachability:** argued through code comparison rather than measured with a user study, so C2 is a supported claim, not a demonstrated one.

**Deferred:**

- **Key-value sort (and `argsort`):** the most valuable extension, since JS has no fast built-in key-value sort, but no claim depends on it. The sort is already stable, so it's planned if time allows. The benchmark would compare against an index sort with a comparator