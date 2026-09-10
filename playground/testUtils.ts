// Shared utils for tests/benchmarks.

export const MIN_LOG2 = 16;
export const MAX_LOG2 = 26;

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
