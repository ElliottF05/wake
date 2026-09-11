import type { Wake } from "../src/index";
import { randomInput } from "./randomInput";
import { formatRow, MAX_LOG2, MIN_LOG2, measureMedianMs } from "./testUtils";

const PERF_WARMUP = 5;
const PERF_ITERS = 32;
const PERF_BATCHES = 9;

export async function testReducePerformance(wake: Wake): Promise<void> {
  console.log("Beginning reduce performance tests:");

  const perfRows: string[] = [];
  for (let log2 = MIN_LOG2; log2 <= MAX_LOG2; log2++) {
    console.log(`measuring 2^${log2}...`);
    const n = 2 ** log2;
    const input = wake.buffer(randomInput("f32", n));
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
    const input = wake.buffer(randomInput("f32", n));
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
