import type { Wake } from "../src/index";
import { formatRow, MAX_LOG2, MIN_LOG2, measureMedianMs } from "./testUtils";

// These tests are auxiliary tests for determining useful information, outside
// of the scope of testing correctness/performance of WAKE itself.

// JS CPU reduction baseline
export async function testJsPerformance(): Promise<void> {
  console.log("Beginning JS CPU baseline:");

  const JS_BATCHES = 5;
  const JS_ITERS = 3;
  const jsData = new Float32Array(2 ** MAX_LOG2);
  for (let i = 0; i < jsData.length; i++) {
    jsData[i] = Math.random() * 2 - 1;
  }

  let jsSink = 0;
  const jsRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring js 2^${log2}...`);
    const n = 2 ** log2;

    const runOne = () => {
      let s = 0;
      for (let i = 0; i < n; i++) {
        s += jsData[i];
      }
      jsSink += s;
    };

    const ms = await measureMedianMs({
      warmup: JS_ITERS,
      batches: JS_BATCHES,
      iters: JS_ITERS,
      run: runOne,
    });

    const gbps = (n * 4) / (ms * 1e6);
    jsRows.push(formatRow(log2, n, ms, gbps));
  }
  console.log(`JS CPU sum baseline (sink=${jsSink}):\n${jsRows.join("\n")}`);
}

// Copy-kernel bandwidth: a simple dst[i] = src[i] kernel to get an idea of the
// max possible memory bandwidth on this system
export async function testCopyKernel(wake: Wake): Promise<void> {
  console.log("Beginning copy kernel bandwidth test:");

  const WARMUP = 5;
  const ITERS = 32;
  const BATCHES = 9;
  const WORKGROUP_SIZE = 256;

  const data = new Float32Array(2 ** MAX_LOG2);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.random() * 2 - 1;
  }

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

    const ms = await measureMedianMs({
      warmup: WARMUP,
      batches: BATCHES,
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

// wake.empty() cost in isolation
export async function testEmptyBuffer(wake: Wake): Promise<void> {
  console.log("Beginning wake.empty() benchmark:");
  const WARMUP = 5;
  const ITERS = 32;
  const BATCHES = 9;
  const emptyRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring empty 2^${log2}...`);
    const n = 2 ** log2;
    const allocate = () => {
      wake.empty(n, Float32Array).destroy();
    };
    const ms = await measureMedianMs({
      warmup: WARMUP,
      batches: BATCHES,
      iters: ITERS,
      run: allocate,
    });
    emptyRows.push(`2^${log2} (${n}): ${ms.toFixed(3)} ms`);
  }
  console.log(`wake.empty() benchmark:\n${emptyRows.join("\n")}`);
}
