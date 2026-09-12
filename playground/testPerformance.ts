import type { WakeBuffer, WakeSupportedArray } from "../src/buffer";
import type { Wake } from "../src/index";
import { randomInput } from "./randomInput";

// ===== shared measurement helpers =====

export const MIN_LOG2 = 16;
export const MAX_LOG2 = 28;

export async function measureMedianMs(opts: {
  warmup: number;
  batches: number;
  iters: number;
  run: () => unknown;
  drain?: () => Promise<void>;
}): Promise<number> {
  const { warmup, batches, iters, run, drain } = opts;
  for (let i = 0; i < warmup; i++) {
    await run();
  }
  if (drain) {
    await drain();
  }
  const perCall: number[] = [];
  for (let b = 0; b < batches; b++) {
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) {
      await run();
    }
    if (drain) {
      await drain();
    }
    perCall.push((performance.now() - t0) / iters);
  }
  perCall.sort((a, b) => a - b);
  return perCall[Math.floor(perCall.length / 2)];
}

export function formatRow(log2: number, n: number, ms: number, gbps: number): string {
  return `2^${log2} (${n}): ${ms.toFixed(3)} ms, ${gbps.toFixed(5)} GB/s`;
}

// ===== WAKE primitive performance sweeps =====

const PERF_WARMUP = 5;
const PERF_ITERS = 32;
const PERF_BATCHES = 1;

// `bytesPerElement` is the total bytes read *and* written per element across
async function runPerfSweep(
  wake: Wake,
  name: string,
  bytesPerElement: number,
  run: (input: WakeBuffer<WakeSupportedArray>) => Promise<void>,
): Promise<void> {
  console.log(`Beginning ${name} performance tests:`);

  const perfRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring 2^${log2}...`);
    const n = 2 ** log2;
    const input = wake.buffer(randomInput("f32", n));
    const ms = await measureMedianMs({
      warmup: PERF_WARMUP,
      batches: PERF_BATCHES,
      iters: PERF_ITERS,
      run: () => run(input),
      drain: () => wake.device.queue.onSubmittedWorkDone(),
    });
    input.destroy();
    const gbps = (bytesPerElement * n) / (ms * 1e6);
    perfRows.push(formatRow(log2, n, ms, gbps));
  }

  console.log(`${name} performance results:\n${perfRows.join("\n")}`);
}

export async function testReducePerformance(wake: Wake): Promise<void> {
  // reads n elements, writes a single (negligible) output
  await runPerfSweep(wake, "Reduce", 4, async (input) => {
    (await wake.reduce(input, "sum")).destroy();
  });
}

export async function testScanPerformance(wake: Wake): Promise<void> {
  // reads input twice (phase 1's reduction, phase 3's re-derived local scan) and writes
  // output once (phase 3), so 3n elements moved total rather than reduce's n
  await runPerfSweep(wake, "Scan", 12, async (input) => {
    (await wake.scan(input, "sum", true)).destroy();
  });
}
