import type { Wake } from "../src/index";
import { benchCpu, benchGpu, type Row, testCorrectness } from "./harness";
import { randomArray } from "./inputs";

// phase 1 reads n elements; phase 2 reads and writes n elements
const BYTES_PER_ELEMENT = 12;

// u32 addition wraps mod 2^32 in WGSL, so wrapping here matches exactly, and wrapping
// addition is associative so the kernel's summation order does not matter.
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
    toBuffer: (w, data) => w.buffer(data),
    run: (input) => wake.scan(input),
    reference: jsScan,
  });
}

export function benchScanGpu(wake: Wake, sizes: number[], trials: number): Promise<Row[]> {
  return benchGpu<"u32">(wake, sizes, trials, {
    label: "wake:scan",
    bytesPerElement: BYTES_PER_ELEMENT,
    makeInput: (n) => randomArray("u32", n),
    toBuffer: (w, data) => w.buffer(data),
    run: (input) => wake.scan(input),
  });
}

export function benchScanCpu(sizes: number[], trials: number): Row[] {
  return benchCpu<"u32">(sizes, trials, {
    label: "js:scan",
    bytesPerElement: BYTES_PER_ELEMENT,
    makeInput: (n) => randomArray("u32", n),
    run: jsScan,
  });
}
