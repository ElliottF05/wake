import type { Wake } from "../src/index";
import { randomInput } from "./randomInput";
import { formatRow, MAX_LOG2, MIN_LOG2, measureMs } from "./testPerformance";

// These tests are auxiliary tests for determining useful information, outside
// of the scope of testing correctness/performance of WAKE itself.

const JS_ITERS = 3;

// JS CPU reduction (sum) baseline
export async function testJsReducePerformance(): Promise<void> {
  console.log("Beginning JS CPU reduce baseline:");

  const jsData = randomInput("f32", 2 ** MAX_LOG2);

  let jsSink = 0;
  const jsRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring js reduce 2^${log2}...`);
    const n = 2 ** log2;

    const runOne = () => {
      let s = 0;
      for (let i = 0; i < n; i++) {
        s += jsData[i];
      }
      jsSink += s;
    };

    const ms = await measureMs({
      warmup: JS_ITERS,
      iters: JS_ITERS,
      run: runOne,
    });

    const gbps = (n * 4) / (ms * 1e6);
    jsRows.push(formatRow(log2, n, ms, gbps));
  }
  console.log(`JS CPU reduce baseline (sink=${jsSink}):\n${jsRows.join("\n")}`);
}

// JS CPU scan (inclusive cumulative sum) baseline
export async function testJsScanPerformance(): Promise<void> {
  console.log("Beginning JS CPU scan baseline:");

  const jsData = randomInput("f32", 2 ** MAX_LOG2);
  const jsOutput = new Float32Array(2 ** MAX_LOG2);

  let jsSink = 0;
  const jsRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring js scan 2^${log2}...`);
    const n = 2 ** log2;

    const runOne = () => {
      let acc = 0;
      for (let i = 0; i < n; i++) {
        acc += jsData[i];
        jsOutput[i] = acc;
      }
      jsSink += jsOutput[n - 1];
    };

    const ms = await measureMs({
      warmup: JS_ITERS,
      iters: JS_ITERS,
      run: runOne,
    });

    // scan reads n elements and writes n elements, unlike reduce's read-only pass
    const gbps = (2 * n * 4) / (ms * 1e6);
    jsRows.push(formatRow(log2, n, ms, gbps));
  }
  console.log(`JS CPU scan baseline (sink=${jsSink}):\n${jsRows.join("\n")}`);
}

// Copy-kernel bandwidth: a simple dst[i] = src[i] kernel to get an idea of the
// max possible memory bandwidth on this system
export async function testCopyKernel(wake: Wake): Promise<void> {
  console.log("Beginning copy kernel bandwidth test:");

  const WARMUP = 5;
  const ITERS = 32;
  const WORKGROUP_SIZE = 256;

  const data = randomInput("f32", 2 ** MAX_LOG2);

  const copyModule = wake.device.createShaderModule({
    label: "wake:playground:copy",
    code: `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let n = arrayLength(&src);
  let stride = nwg.x * ${WORKGROUP_SIZE}u;
  for (var i = gid.x; i < n; i += stride) {
    dst[i] = src[i];
  }
}
`,
  });
  const copyPipeline = await wake.device.createComputePipelineAsync({
    label: "wake:playground:copy",
    layout: "auto",
    compute: { module: copyModule },
  });

  const copyRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring copy 2^${log2}...`);
    const n = 2 ** log2;
    const src = wake.device.createBuffer({
      label: "wake:playground:copy-src",
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    wake.device.queue.writeBuffer(src, 0, data.subarray(0, n));
    const dst = wake.device.createBuffer({
      label: "wake:playground:copy-dst",
      size: n * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    const bindGroup = wake.device.createBindGroup({
      layout: copyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
      ],
    });
    const workgroups = Math.min(Math.ceil(n / WORKGROUP_SIZE), 65535);
    const submitCopy = () => {
      const encoder = wake.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(copyPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      wake.device.queue.submit([encoder.finish()]);
    };

    const ms = await measureMs({
      warmup: WARMUP,
      iters: ITERS,
      run: submitCopy,
      drain: () => wake.device.queue.onSubmittedWorkDone(),
    });

    src.destroy();
    dst.destroy();
    const gbps = (2 * n * 4) / (ms * 1e6);
    copyRows.push(formatRow(log2, n, ms, gbps));
  }
  console.log(`Copy kernel bandwidth:\n${copyRows.join("\n")}`);
}

// Copy-kernel bandwidth, measured with GPU timestamp-query instead of wall-clock. This
// needs its own device requesting the "timestamp-query" feature — `wake`'s device wasn't
// created with it — so it's fully self-contained rather than taking `wake` as a param.
// Kept alongside testCopyKernel (not replacing it) so the two methods' numbers and
// required sample counts can be compared directly.
export async function testCopyKernelTimestamp(): Promise<void> {
  console.log("Beginning copy kernel bandwidth test (timestamp-query):");

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    console.log("no WebGPU adapter available");
    return;
  }
  if (!adapter.features.has("timestamp-query")) {
    console.log("this adapter does not support timestamp-query, skipping");
    return;
  }
  const device = await adapter.requestDevice({
    label: "wake:playground:copy-timestamp",
    requiredFeatures: ["timestamp-query"],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  const WARMUP = 5;
  const ITERS = 20;
  const WORKGROUP_SIZE = 256;

  const data = randomInput("f32", 2 ** MAX_LOG2);

  const copyModule = device.createShaderModule({
    label: "wake:playground:copy-timestamp",
    code: `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let n = arrayLength(&src);
  let stride = nwg.x * ${WORKGROUP_SIZE}u;
  for (var i = gid.x; i < n; i += stride) {
    dst[i] = src[i];
  }
}
`,
  });
  const copyPipeline = await device.createComputePipelineAsync({
    label: "wake:playground:copy-timestamp",
    layout: "auto",
    compute: { module: copyModule },
  });

  // One (begin, end) pair, reused every iteration: resolve + read back, then dispatch again.
  const querySet = device.createQuerySet({
    label: "wake:playground:copy-timestamps",
    type: "timestamp",
    count: 2,
  });
  const resolveBuffer = device.createBuffer({
    label: "wake:playground:copy-timestamps-resolve",
    size: 2 * 8, // 2 x u64 nanosecond timestamps
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    label: "wake:playground:copy-timestamps-readback",
    size: 2 * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const copyRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring copy 2^${log2}...`);
    const n = 2 ** log2;
    const src = device.createBuffer({
      label: "wake:playground:copy-src",
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(src, 0, data.subarray(0, n));
    const dst = device.createBuffer({
      label: "wake:playground:copy-dst",
      size: n * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    const bindGroup = device.createBindGroup({
      layout: copyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
      ],
    });
    const workgroups = Math.min(Math.ceil(n / WORKGROUP_SIZE), 65535);

    // Runs the kernel once and returns the GPU's own measurement of its execution time,
    // in nanoseconds — no wall-clock, no onSubmittedWorkDone() round trip.
    const runOnceNs = async (): Promise<number> => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass({
        timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
      });
      pass.setPipeline(copyPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, readbackBuffer.size);
      device.queue.submit([encoder.finish()]);

      await readbackBuffer.mapAsync(GPUMapMode.READ);
      const [begin, end] = new BigInt64Array(readbackBuffer.getMappedRange().slice(0));
      readbackBuffer.unmap();
      return Number(end - begin);
    };

    for (let i = 0; i < WARMUP; i++) {
      await runOnceNs();
    }
    const samplesNs: number[] = [];
    for (let i = 0; i < ITERS; i++) {
      samplesNs.push(await runOnceNs());
    }
    const ns = samplesNs.reduce((acc, x) => acc + x) / samplesNs.length;

    src.destroy();
    dst.destroy();
    // bytes / ns is exactly GB/s (both the 1e9 in "giga" and the 1e-9 in "nano" cancel)
    const gbps = (2 * n * 4) / ns;
    copyRows.push(formatRow(log2, n, ns / 1e6, gbps));
  }

  device.destroy();
  console.log(`Copy kernel bandwidth (timestamp-query):\n${copyRows.join("\n")}`);
}

// wake.empty() cost in isolation
export async function testEmptyBuffer(wake: Wake): Promise<void> {
  console.log("Beginning wake.empty() benchmark:");
  const WARMUP = 5;
  const ITERS = 32;
  const emptyRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring empty 2^${log2}...`);
    const n = 2 ** log2;
    const allocate = () => {
      wake.empty(n, Float32Array).destroy();
    };
    const ms = await measureMs({
      warmup: WARMUP,
      iters: ITERS,
      run: allocate,
    });
    emptyRows.push(`2^${log2} (${n}): ${ms.toFixed(3)} ms`);
  }
  console.log(`wake.empty() benchmark:\n${emptyRows.join("\n")}`);
}
