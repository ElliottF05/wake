import type { Wake } from "../src/index";
import { type BenchResult, benchCpu, benchGpu, testCorrectness } from "./harness";
import { randomArray } from "./inputs";

// phase 1 reads n elements; phase 2 reads and writes n elements
const GPU_BYTES_PER_ELEMENT = 4 * 3;
// one sequential pass: reads n elements, writes n elements
const CPU_BYTES_PER_ELEMENT = 4 * 2;

// handles wrapping
export function jsScan(input: Uint32Array): Uint32Array {
  const output = new Uint32Array(input.length);
  let acc = 0;
  for (let i = 0; i < input.length; i++) {
    output[i] = acc;
    acc = (acc + input[i]) >>> 0;
  }
  return output;
}

export function testScanCorrectness(wake: Wake, sizes: number[]): Promise<void> {
  return testCorrectness<"u32">(wake, sizes, {
    label: "scan",
    makeInput: (n) => randomArray("u32", n),
    run: (input) => wake.scan(input),
    reference: jsScan,
  });
}

export function benchScanGpu(wake: Wake, sizes: number[], trials: number): Promise<BenchResult[]> {
  return benchGpu<"u32">(wake, sizes, trials, {
    label: "wake:scan",
    bytesPerElement: GPU_BYTES_PER_ELEMENT,
    makeInput: (n) => randomArray("u32", n),
    run: (input) => wake.scan(input),
  });
}

export function benchScanCpu(sizes: number[], trials: number): BenchResult[] {
  return benchCpu<"u32">(sizes, trials, {
    label: "js:scan",
    bytesPerElement: CPU_BYTES_PER_ELEMENT,
    makeInput: (n) => randomArray("u32", n),
    run: jsScan,
  });
}
