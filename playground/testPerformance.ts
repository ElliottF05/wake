import type { WakeBuffer, WakeSupportedArray } from "../src/buffer";
import type { Wake } from "../src/index";
import { randomInput } from "./randomInput";

// ===== shared measurement helpers =====

export async function measureMs(opts: {
  warmup: number;
  iters: number;
  run: () => unknown;
  drain?: () => Promise<void>;
}): Promise<number> {
  const { warmup, iters, run, drain } = opts;
  for (let i = 0; i < warmup; i++) {
    await run();
  }
  if (drain) {
    await drain();
  }
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    await run();
  }
  if (drain) {
    await drain();
  }
  return (performance.now() - t0) / iters;
}

export function formatRow(log2: number, n: number, ms: number, gbps: number): string {
  return `2^${log2} (${n}): ${ms.toFixed(3)} ms, ${gbps.toFixed(5)} GB/s`;
}

// ===== WAKE primitive performance sweeps =====

const PERF_WARMUP = 5;
const PERF_ITERS = 16;

// `bytesPerElement` is the total bytes read *and* written per element across
async function runPerfSweep(
  wake: Wake,
  minLog2: number,
  maxLog2: number,
  name: string,
  bytesPerElement: number,
  run: (input: WakeBuffer<WakeSupportedArray>) => Promise<void>,
): Promise<void> {
  console.log(`Beginning ${name} performance tests:`);

  const perfRows: string[] = [];
  for (let log2 = minLog2; log2 <= maxLog2; log2++) {
    console.log(`measuring 2^${log2}...`);
    const n = 2 ** log2;
    const input = wake.buffer(randomInput("f32", n));
    const ms = await measureMs({
      warmup: PERF_WARMUP,
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

export async function testReducePerformance(
  wake: Wake,
  minLog2: number,
  maxLog2: number,
): Promise<void> {
  // reads n elements, writes a single (negligible) output
  await runPerfSweep(wake, minLog2, maxLog2, "Reduce", 4, async (input) => {
    (await wake.reduce(input, "sum")).destroy();
  });
}

export async function testScanPerformance(
  wake: Wake,
  minLog2: number,
  maxLog2: number,
): Promise<void> {
  // reads input twice (phase 1's reduction, phase 3's re-derived local scan) and writes
  // output once (phase 3), so 3n elements moved total rather than reduce's n
  await runPerfSweep(wake, minLog2, maxLog2, "Scan", 12, async (input) => {
    (await wake.scan(input, "sum")).destroy();
  });
}
