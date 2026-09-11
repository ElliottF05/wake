import type { Wake } from "../src/index";
import type { WgslDtype } from "../src/ops";
import { formatRow, MAX_LOG2, MIN_LOG2, measureMedianMs } from "./testUtils";

// ===== correctness: input generation =====

const DTYPES = ["f32", "u32", "i32"] as const;

// 0, 1, small values, powers of two and +/-1 around them, plus one large case
// biome-ignore format: keep on one line
const INPUT_SIZES = [0, 1, 2, 3, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537, 1 << 20];

function randomInput(dtype: WgslDtype, n: number): Float32Array | Uint32Array | Int32Array {
  if (dtype === "f32") {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      a[i] = Math.random() * 2 - 1;
    }
    return a;
  }
  const a = dtype === "u32" ? new Uint32Array(n) : new Int32Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = Math.random() * 2 ** 32; // typed-array assignment wraps to the full u32/i32 range
  }
  return a;
}

// Used on sum operations to avoid overflow
function randomSumInput(dtype: WgslDtype, n: number): Float32Array | Uint32Array | Int32Array {
  const values = Array.from({ length: n }, () =>
    dtype === "f32" ? Math.random() * 1000 : Math.floor(Math.random() * 1001),
  );
  if (dtype === "f32") return new Float32Array(values);
  if (dtype === "u32") return new Uint32Array(values);
  return new Int32Array(values);
}

// ===== correctness: check + report =====

let checksRun = 0;
const failures: string[] = [];

function check(
  name: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  approx = false,
): void {
  checksRun++;
  if (actual.length !== expected.length) {
    failures.push(`${name}: length ${actual.length}, expected ${expected.length}`);
    return;
  }
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const ok = approx ? Math.abs(a - e) <= 1e-3 * Math.max(1, Math.abs(e)) : a === e;
    if (!ok) {
      failures.push(`${name}: index ${i}, got ${a}, expected ${e}`);
      return;
    }
  }
}

// ===== correctness: JS reference implementations =====

type Op = "sum" | "min" | "max";

const F32_MAX = new Float32Array(new Uint32Array([0x7f7fffff]).buffer)[0];
const IDENTITY: Record<Op, Record<WgslDtype, number>> = {
  sum: { f32: 0, u32: 0, i32: 0 },
  min: { f32: F32_MAX, u32: 0xffffffff, i32: 2147483647 },
  max: { f32: -F32_MAX, u32: 0, i32: -2147483648 },
};

const COMBINE: Record<Op, (a: number, b: number) => number> = {
  sum: (a, b) => a + b,
  min: (a, b) => Math.min(a, b),
  max: (a, b) => Math.max(a, b),
};

function jsReduce(input: number[], op: Op, dtype: WgslDtype): number[] {
  if (input.length === 0) {
    // reduce(empty) returns a zero-filled buffer, not the op's true identity — see reduce.ts
    return [0];
  }
  let acc = IDENTITY[op][dtype];
  for (const x of input) {
    acc = COMBINE[op](acc, x);
  }
  return [acc];
}

function jsScan(input: number[], op: Op, inclusive: boolean, dtype: WgslDtype): number[] {
  const out: number[] = [];
  let acc = IDENTITY[op][dtype];
  for (const x of input) {
    if (inclusive) {
      acc = COMBINE[op](acc, x);
      out.push(acc);
    } else {
      out.push(acc);
      acc = COMBINE[op](acc, x);
    }
  }
  return out;
}

function jsSort(input: number[]): number[] {
  return [...input].sort((a, b) => a - b);
}

// ===== testCorrectness =====

export async function testCorrectness(wake: Wake): Promise<void> {
  // buffer API round-trips
  {
    const f = new Float32Array([1, 2, 3, 4, 5]);
    check("buffer round-trip", await wake.buffer(f).read(), f);

    const u = new Uint32Array([10, 20, 30, 40]);
    const written = wake.buffer(new Uint32Array(4));
    written.write(u);
    check("buffer write", await written.read(), u);

    const i = new Int32Array([-1, -2, -3]);
    const raw = wake.device.createBuffer({
      size: i.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const wrapped = wake.wrap(raw, Int32Array);
    wrapped.write(i);
    check("wrap external GPUBuffer", await wrapped.read(), i);
  }

  // validate each (primitive, op, dtype, size) against the corresponding JS 
  // implementation
  for (const dtype of DTYPES) {
    for (const n of INPUT_SIZES) {
      const input = randomInput(dtype, n);

      for (const op of ["sum", "min", "max"] as const) {
        const approx = op === "sum" && dtype === "f32";
        const opInput = op === "sum" ? randomSumInput(dtype, n) : input;
        const ref = [...opInput];

        const rin = wake.buffer(opInput);
        const rout = await wake.reduce(rin, op);
        check(`reduce/${op}/${dtype}/n=${n}`, await rout.read(), jsReduce(ref, op, dtype), approx);
        rin.destroy();
        rout.destroy();

        for (const inclusive of [true, false]) {
          const sin = wake.buffer(opInput);
          const sout = await wake.scan(sin, op, inclusive);
          check(
            `scan/${op}/${inclusive ? "inc" : "exc"}/${dtype}/n=${n}`,
            await sout.read(),
            jsScan(ref, op, inclusive, dtype),
            approx,
          );
          sin.destroy();
          sout.destroy();
        }
      }

      const kbuf = wake.buffer(input);
      await wake.radixSort(kbuf);
      check(`radixSort/${dtype}/n=${n}`, await kbuf.read(), jsSort([...input]));
      kbuf.destroy();
    }
  }

  // report correctness results
  console.log(
    failures.length === 0
      ? `all ${checksRun} correctness checks passed`
      : `${failures.length}/${checksRun} correctness checks failed:\n${failures.join("\n")}`,
  );
  checksRun = 0;
  failures.length = 0;
}

// ===== performance tests =====

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
