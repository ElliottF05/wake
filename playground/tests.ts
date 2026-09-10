import type { Wake } from "../src/index";
import type { WgslDtype } from "../src/ops";
import { formatRow, MAX_LOG2, MIN_LOG2, measureMedianMs } from "./testUtils";

export async function testCorrectness(wake: Wake): Promise<void> {
  let passed = 0;
  let failed = 0;

  function report(name: string, ok: boolean, reason?: string): void {
    if (ok) {
      passed++;
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

  function checkArrayApprox(
    name: string,
    actual: ArrayLike<number>,
    expected: ArrayLike<number>,
  ): void {
    if (actual.length !== expected.length) {
      report(name, false, `expected length ${expected.length}, got ${actual.length}`);
      return;
    }
    for (let i = 0; i < expected.length; i++) {
      if (Math.abs(actual[i] - expected[i]) > 1e-6) {
        report(
          name,
          false,
          `expected [${Array.from(expected)}], got [${Array.from(actual)}] (first mismatch at index ${i})`,
        );
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
  const F32_MAX = new Float32Array(new Uint32Array([0x7f7fffff]).buffer)[0];

  const reductions: Record<
    "sum" | "min" | "max",
    { f: (a: number, b: number) => number; identity: Record<WgslDtype, number> }
  > = {
    sum: {
      f: (a, b) => a + b,
      identity: { f32: 0, u32: 0, i32: 0 },
    },
    min: {
      f: (a, b) => Math.min(a, b),
      identity: { f32: F32_MAX, u32: 0xffffffff, i32: 2147483647 },
    },
    max: {
      f: (a, b) => Math.max(a, b),
      identity: { f32: -F32_MAX, u32: 0, i32: -2147483648 },
    },
  };

  const datasets = [
    { dtype: "f32" as const, data: new Float32Array([3.5, 1.25, 4, 1, 5, 9.5, 2, 6]) },
    { dtype: "u32" as const, data: new Uint32Array([3, 1, 4, 1, 5, 9, 2, 6]) },
    { dtype: "i32" as const, data: new Int32Array([3, -1, 4, -1, 5, -9, 2, -6]) },
  ];

  for (const { dtype, data } of datasets) {
    for (const op of ["sum", "min", "max"] as const) {
      const { f, identity } = reductions[op];
      const expected = Array.from(data).reduce((acc, x) => f(acc, x), identity[dtype]);
      const result = await (await wake.reduce(wake.buffer(data), op)).read();
      checkScalar(`reduce correctness (${op}, ${dtype})`, result, expected);
    }
  }

  // 5. test scan correctness against a JS implementation for each (op, dtype, inclusive)
  function jsScan(
    data: ArrayLike<number>,
    f: (a: number, b: number) => number,
    identity: number,
    inclusive: boolean,
  ): number[] {
    const result: number[] = [];
    let acc = identity;
    for (let i = 0; i < data.length; i++) {
      if (inclusive) {
        acc = f(acc, data[i]);
        result.push(acc);
      } else {
        result.push(acc);
        acc = f(acc, data[i]);
      }
    }
    return result;
  }

  for (const { dtype, data } of datasets) {
    for (const op of ["sum", "min", "max"] as const) {
      const { f, identity } = reductions[op];
      for (const inclusive of [true, false]) {
        const expected = jsScan(data, f, identity[dtype], inclusive);
        const result = await (await wake.scan(wake.buffer(data), op, inclusive)).read();
        const mode = inclusive ? "inclusive" : "exclusive";
        checkArrayApprox(`scan correctness (${op}, ${dtype}, ${mode})`, result, expected);
      }
    }
  }

  console.log(`Correctness test results: ${passed} passed, ${failed} failed`);
}

const PERF_WARMUP = 5;
const PERF_ITERS = 32;
const PERF_BATCHES = 9;

export async function testReducePerformance(wake: Wake): Promise<void> {
  console.log("Beginning reduce performance tests:");

  const perfRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring 2^${log2}...`);
    const n = 2 ** log2;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    const input = wake.buffer(data);
    const ms = await measureMedianMs({
      warmup: PERF_WARMUP,
      batches: PERF_BATCHES,
      iters: PERF_ITERS,
      run: () => wake.reduce(input, "sum"),
      drain: () => wake.device.queue.onSubmittedWorkDone(),
    });
    input.destroy();
    const gbps = (n * 4) / (ms * 1e6);
    perfRows.push(formatRow(log2, n, ms, gbps));
  }

  console.log(`Reduce performance results:\n${perfRows.join("\n")}`);
}

export async function testScanPerformance(wake: Wake): Promise<void> {
  console.log("Beginning scan performance tests:");

  const perfRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring 2^${log2}...`);
    const n = 2 ** log2;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    const input = wake.buffer(data);
    const ms = await measureMedianMs({
      warmup: PERF_WARMUP,
      batches: PERF_BATCHES,
      iters: PERF_ITERS,
      run: async () => {
        (await wake.scan(input, "sum", true)).destroy();
      },
      drain: () => wake.device.queue.onSubmittedWorkDone(),
    });
    input.destroy();
    // scan reads input twice (phase 1's reduction, phase 3's re-derived local scan) and
    // writes output once (phase 3), so 3n elements moved total rather than reduce's n.
    const gbps = (3 * n * 4) / (ms * 1e6);
    perfRows.push(formatRow(log2, n, ms, gbps));
  }

  console.log(`Scan performance results:\n${perfRows.join("\n")}`);
}
