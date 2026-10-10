import type { WakeArrays, WakeBuffer, WakeDtype } from "../src/buffer";
import type { Wake } from "../src/index";

export type Row = {
  label: string;
  n: number;
  meanMs: number;
  minMs: number;
  kernelMeanMs?: number;
  kernelMinMs?: number;
  gbps?: number;
};

type CorrectnessCase<D extends WakeDtype> = {
  label: string;
  makeInput: (n: number) => WakeArrays[D];
  toBuffer: (wake: Wake, data: WakeArrays[D]) => WakeBuffer<D>;
  run: (input: WakeBuffer<D>) => WakeBuffer<D>;
  reference: (input: WakeArrays[D]) => WakeArrays[D];
};

type GpuBench<D extends WakeDtype> = {
  label: string;
  bytesPerElement?: number;
  makeInput: (n: number) => WakeArrays[D];
  toBuffer: (wake: Wake, data: WakeArrays[D]) => WakeBuffer<D>;
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
    const input = test.toBuffer(wake, data);
    const output = test.run(input);
    const actual = await output.read();

    let failures = 0;
    let firstFailure = -1;
    for (let i = 0; i < n; i++) {
      if (actual[i] !== expected[i]) {
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
): Promise<Row[]> {
  const rows: Row[] = [];

  for (const n of sizes) {
    const data = bench.makeInput(n);
    const input = bench.toBuffer(wake, data);

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

    const kernelMeanMs = mean(kernelMs);
    rows.push({
      label: bench.label,
      n,
      meanMs: round3(mean(wallMs)),
      minMs: round3(Math.min(...wallMs)),
      kernelMeanMs: round3(kernelMeanMs),
      kernelMinMs: round3(Math.min(...kernelMs)),
      gbps: gbps(bench.bytesPerElement, n, kernelMeanMs),
    });
  }

  return rows;
}

export function benchCpu<D extends WakeDtype>(
  sizes: number[],
  trials: number,
  bench: CpuBench<D>,
): Row[] {
  const rows: Row[] = [];
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

    const meanMs = mean(wallMs);
    rows.push({
      label: bench.label,
      n,
      meanMs: round3(meanMs),
      minMs: round3(Math.min(...wallMs)),
      gbps: gbps(bench.bytesPerElement, n, meanMs),
    });
  }

  // keeps the runs from being optimized away
  console.debug(`${bench.label}: sink`, sink);
  return rows;
}

// `gbps` is computed from kernel time for GPU rows and wall time for CPU rows.
export function reportRows(rows: Row[]): void {
  console.table(rows);

  const header = "label,n,meanMs,minMs,kernelMeanMs,kernelMinMs,gbps";
  const csv = rows.map((row) =>
    [
      row.label,
      row.n,
      row.meanMs,
      row.minMs,
      row.kernelMeanMs ?? "",
      row.kernelMinMs ?? "",
      row.gbps ?? "",
    ].join(","),
  );
  console.log([header, ...csv].join("\n"));
}
