import { Wake } from "../src/index";

const wake = await Wake.create();

let passed = 0;
let failed = 0;

function report(name: string, ok: boolean, reason?: string): void {
  if (ok) {
    passed++;
    console.log(`${name}: passed`);
  } else {
    failed++;
    console.log(`${name}: failed, ${reason}`);
  }
}

function checkArray(name: string, actual: ArrayLike<number>, expected: ArrayLike<number>): void {
  if (actual.length !== expected.length) {
    report(name, false, `expected length ${expected.length}, got ${actual.length}`);
    return;
  }
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) {
      report(name, false, `expected [${Array.from(expected)}], got [${Array.from(actual)}]`);
      return;
    }
  }
  report(name, true);
}

function checkScalar(name: string, actual: ArrayLike<number>, expected: number): void {
  if (actual.length !== 1) {
    report(name, false, `expected a length-1 result, got length ${actual.length}`);
    return;
  }
  if (Math.abs(actual[0] - expected) > 1e-6) {
    report(name, false, `expected ${expected}, got ${actual[0]}`);
    return;
  }
  report(name, true);
}

// 1. create from data, read it back
{
  const name = "buffer round-trip";
  const input = new Float32Array([1, 2, 3, 4, 5]);
  checkArray(name, await wake.buffer(input).read(), input);
}

// 2. create a buffer, overwrite it with new data, read back
{
  const name = "buffer write";
  const input = new Uint32Array([10, 20, 30, 40]);
  const buf = wake.buffer(new Uint32Array([0, 0, 0, 0]));
  buf.write(input);
  checkArray(name, await buf.read(), input);
}

// 3. wrap an externally created GPUBuffer, then write to it and read back
{
  const name = "wrap external GPUBuffer";
  const input = new Int32Array([-1, -2, -3]);
  const raw = wake.device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const wrapped = wake.wrap(raw, Int32Array);
  wrapped.write(input);
  checkArray(name, await wrapped.read(), input);
}

// 4. test correctness against JS implementation for each (op, dtype)
const reductions = {
  sum: { f: (a: number, b: number) => a + b, identity: 0 },
  min: { f: (a: number, b: number) => Math.min(a, b), identity: Number.POSITIVE_INFINITY },
  max: { f: (a: number, b: number) => Math.max(a, b), identity: Number.NEGATIVE_INFINITY },
};

const datasets = [
  { dtype: "f32", data: new Float32Array([3.5, 1.25, 4, 1, 5, 9.5, 2, 6]) },
  { dtype: "u32", data: new Uint32Array([3, 1, 4, 1, 5, 9, 2, 6]) },
  { dtype: "i32", data: new Int32Array([3, -1, 4, -1, 5, -9, 2, -6]) },
];

for (const { dtype, data } of datasets) {
  for (const op of ["sum", "min", "max"] as const) {
    const { f, identity } = reductions[op];
    const expected = Array.from(data).reduce((acc, x) => f(acc, x), identity);
    const result = await (await wake[op](wake.buffer(data))).read();
    checkScalar(`correctness (${op}, ${dtype})`, result, expected);
  }
}

console.log(`Correctness test results: ${passed} passed, ${failed} failed`);
console.log(`Beginning performance tests:`)

// 5. performance testing on increasing input sizes.
const MIN_LOG2 = 16;
const MAX_LOG2 = 28;
const WARMUP = 5;
const ITERS = 100;
const BATCHES = 5;

for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
  const n = 2 ** log2;
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    data[i] = Math.random() * 2 - 1;
  }
  const input = wake.buffer(data);

  for (let i = 0; i < WARMUP; i++) {
    await wake.sum(input);
  }
  await wake.device.queue.onSubmittedWorkDone();

  const perCall: number[] = [];
  for (let b = 0; b < BATCHES; b++) {
    const t0 = performance.now();
    for (let i = 0; i < ITERS; i++) {
      await wake.sum(input);
    }
    await wake.device.queue.onSubmittedWorkDone();
    perCall.push((performance.now() - t0) / ITERS);
  }

  perCall.sort((a, b) => a - b);
  const ms = perCall[Math.floor(perCall.length / 2)];
  const gbps = (n * 4) / (ms * 1e6);
  console.log(`  2^${log2} (${n}): ${ms.toFixed(3)} ms, ${gbps.toFixed(5)} GB/s`);
}
