import type { WakeArrays, WakeBuffer, WakeDtype } from "../src/buffer";
import type { Wake } from "../src/index";

export type BenchResult = {
  label: string;
  n: number;
  bytesPerElement?: number;
  wallMs: number[];
  kernelMs: number[];
};

type CorrectnessCase<D extends WakeDtype> = {
  label: string;
  makeInput: (n: number) => WakeArrays[D];
  run: (input: WakeBuffer<D>) => WakeBuffer<D>;
  reference: (input: WakeArrays[D]) => WakeArrays[D];
  tolerance?: (input: WakeArrays[D]) => number;
};

type GpuBench<D extends WakeDtype> = {
  label: string;
  bytesPerElement?: number;
  makeInput: (n: number) => WakeArrays[D];
  run: (input: WakeBuffer<D>) => WakeBuffer<D>;
};

type CpuBench<D extends WakeDtype> = {
  label: string;
  bytesPerElement?: number;
  makeInput: (n: number) => WakeArrays[D];
  run: (input: WakeArrays[D]) => unknown;
};

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function gbps(bytesPerElement: number | undefined, n: number, ms: number): number | undefined {
  if (bytesPerElement === undefined || ms === 0) {
    return undefined;
  }
  return round3((bytesPerElement * n) / (ms * 1e6));
}

export async function testCorrectness<D extends WakeDtype>(
  wake: Wake,
  sizes: number[],
  test: CorrectnessCase<D>,
): Promise<void> {
  console.log(`${test.label}: correctness`);

  for (const n of sizes) {
    const data = test.makeInput(n);
    const expected = test.reference(data);
    const tolerance = test.tolerance ? test.tolerance(data) : 0;
    const input = wake.buffer(data) as WakeBuffer<D>;
    const output = test.run(input);
    const actual = await output.read();

    let failures = 0;
    let firstFailure = -1;
    for (let i = 0; i < expected.length; i++) {
      if (Math.abs(actual[i] - expected[i]) > tolerance) {
        failures++;
        if (firstFailure === -1) {
          firstFailure = i;
        }
      }
    }

    if (failures === 0) {
      console.log(`  n=${n}: ok`);
    } else {
      console.error(
        `  n=${n}: ${failures} mismatches, first at index ${firstFailure} ` +
          `(expected ${expected[firstFailure]}, got ${actual[firstFailure]})`,
      );
    }

    if (output !== input) {
      output.destroy();
    }
    input.destroy();
  }
}

export async function benchGpu<D extends WakeDtype>(
  wake: Wake,
  sizes: number[],
  trials: number,
  bench: GpuBench<D>,
): Promise<BenchResult[]> {
  const results: BenchResult[] = [];

  for (const n of sizes) {
    const data = bench.makeInput(n);
    const input = wake.buffer(data) as WakeBuffer<D>;

    const warmup = bench.run(input);
    await warmup.read();
    if (warmup !== input) {
      warmup.destroy();
    }

    const wallMs: number[] = [];
    const kernelMs: number[] = [];
    for (let trial = 0; trial < trials; trial++) {
      const start = performance.now();
      input.write(data);
      const output = bench.run(input);
      await output.read();
      wallMs.push(performance.now() - start);
      kernelMs.push(await wake.getLastTimingMs());
      if (output !== input) {
        output.destroy();
      }
    }
    input.destroy();

    results.push({
      label: bench.label,
      n,
      bytesPerElement: bench.bytesPerElement,
      wallMs,
      kernelMs,
    });
  }

  return results;
}

export function benchCpu<D extends WakeDtype>(
  sizes: number[],
  trials: number,
  bench: CpuBench<D>,
): BenchResult[] {
  const results: BenchResult[] = [];
  let sink: unknown;

  for (const n of sizes) {
    const data = bench.makeInput(n);

    sink = bench.run(data);

    const wallMs: number[] = [];
    for (let trial = 0; trial < trials; trial++) {
      const start = performance.now();
      sink = bench.run(data);
      wallMs.push(performance.now() - start);
    }

    results.push({
      label: bench.label,
      n,
      bytesPerElement: bench.bytesPerElement,
      wallMs,
      kernelMs: [],
    });
  }

  // keeps the runs from being optimized away
  console.debug(`${bench.label}: sink`, sink);
  return results;
}

export function reportResults(results: BenchResult[]): void {
  console.table(
    results.map((result) => {
      const hasKernel = result.kernelMs.length > 0;
      const computeMs = hasKernel ? mean(result.kernelMs) : mean(result.wallMs);
      return {
        label: result.label,
        n: result.n,
        meanMs: round3(mean(result.wallMs)),
        minMs: round3(Math.min(...result.wallMs)),
        kernelMeanMs: hasKernel ? round3(mean(result.kernelMs)) : undefined,
        kernelMinMs: hasKernel ? round3(Math.min(...result.kernelMs)) : undefined,
        gbps: gbps(result.bytesPerElement, result.n, computeMs),
      };
    }),
  );

  const header = "label,n,trial,wallMs,kernelMs,bytesPerElement";
  const rows = results.flatMap((result) =>
    result.wallMs.map((wallMs, trial) =>
      [
        result.label,
        result.n,
        trial,
        wallMs,
        result.kernelMs[trial] ?? "",
        result.bytesPerElement ?? "",
      ].join(","),
    ),
  );
  console.log([header, ...rows].join("\n"));
}
