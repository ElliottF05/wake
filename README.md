# WAKE

**WebGPU Approachable Kernel Engine**: approachable GPU computing in the browser, for JavaScript
developers.

WAKE provides JavaScript developers with fast, GPU-accelerated reduction, prefix scan, and sort through
calls that look like ordinary JavaScript. There are no shaders to write and no pipelines,
workgroups, or buffer usage flags to manage.

WAKE was created as part of my undergraduate thesis at Georgia Tech.

## Purpose

GPUs can run highly parallel workloads far faster than CPUs, but using them has traditionally
meant learning a platform-specific programming model such as CUDA or Metal. WebGPU removes the
platform problem, since it runs in any modern browser, but it is still a low-level API where developers
write their own kernels in WGSL and manage GPU resources themselves.

On native platforms, primitives libraries such as Thrust and cuDNN solved this by providing
optimized building blocks that developers call without writing GPU code. WAKE aims to do the same
for WebGPU, with code comparable in simplicity to plain JavaScript and performance faster than
JavaScript above certain input sizes.

## Example

```js
import { Wake } from "wake";

const wake = await Wake.create();

// Sum a Float32Array on the GPU
const [sum] = await wake.buffer(data).sum().read();

// Sort a Float32Array on the GPU
const sorted = await wake.buffer(data).sort().read();
```

Data stays on the GPU between calls, so several primitives can run on the same buffer without
copying the input again:

```js
const values = wake.buffer(data); // copy to the GPU once
const [total] = await values.sum().read(); // only the result (one value) is read back
const [largest] = await values.max().read();
```

## Primitives

| Primitive | Types | Notes |
| :-- | :-- | :-- |
| `sum()`, `min()`, `max()` | u32, f32 | Return a new one-element buffer |
| `scan()` | u32 | Exclusive prefix sum, in place |
| `sort()` | u32, f32 | Ascending, stable, in place |

u32 and f32 buffers are created from `Uint32Array` and `Float32Array`.

## Performance

Measured on an Apple M2 MacBook Air in Chrome, at the largest size tested for each primitive
(2^28 elements for reduce and scan, 2^25 for sort), compared with plain
JavaScript:

| Primitive | Kernel-only speedup | Round-trip speedup |
| :-- | --: | --: |
| Sort (f32) | 61x | 32x |
| Reduce (f32 sum) | 22x | 1.37x |
| Scan (u32) | 9.2x | 0.79x |

*Kernel-only time is GPU execution time alone. Round-trip time also includes copying the input to
the GPU and reading the result back.*

- At large sizes, the reduce and scan kernels reach about 95% (reduce) and up to 90% (scan) of the M2's theoretical peak memory bandwidth.
- Data movement between the CPU and GPU is a large cost. Sort does enough work per element to stay much faster than JavaScript even in round-trip time. Reduce and scan are fastest when data is already on the GPU.
- WAKE beats JavaScript in round-trip time from about 2^15 elements for sort and about
  2^20 for reduce. For kernel-only time, WAKE wins starting from 2^12 to 2^16 elements,
  depending on the primitive.

All results come from a single device, so these numbers may differ on other hardware.

## Requirements

- A browser with WebGPU (tested in Chrome)
- A GPU with the WebGPU `subgroups` and `timestamp-query` features, and a subgroup size of 32.

## Development

WAKE is not yet published as a package. To run the test and benchmark playground locally:

```bash
npm install
npm run dev
```

The playground checks every primitive against a plain JavaScript reference and benchmarks it
across input sizes. Accurate small-size GPU timings need Chrome's
`chrome://flags/#enable-webgpu-developer-features` flag, since timestamp queries are otherwise
rounded to 100 µs.

## Future work

- **More primitives.** Key-value sort, matrix multiplication, etc.
- **Support for subgroup sizes other than 32**, so WAKE can run on more GPUs.

## License

MIT
