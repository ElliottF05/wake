import type { Wake } from "../src/index";
import { type BenchResult, benchCpu, benchGpu, testCorrectness } from "./harness";
import { randomArray } from "./inputs";

// per pass, in bytes per key: count reads keys (4) and writes the histogram (0.5), the
// histogram scan moves 1.5, reorder reads keys (4), reads the histogram (0.5) and writes keys
// (4). 14.5 per pass over 4 passes = 58 total
const GPU_BYTES_PER_ELEMENT = 58;

export function jsSortU32(input: Uint32Array): Uint32Array {
  return Uint32Array.from(input).sort();
}

export function jsSortF32(input: Float32Array): Float32Array {
  return Float32Array.from(input).sort();
}

export async function testSortCorrectness(wake: Wake, sizes: number[]): Promise<void> {
  await testCorrectness<"u32">(wake, sizes, {
    label: "sort:u32",
    makeInput: (n) => randomArray("u32", n),
    run: (keys) => wake.sort(keys),
    reference: jsSortU32,
  });

  await testCorrectness<"f32">(wake, sizes, {
    label: "sort:f32",
    makeInput: (n) => randomArray("f32", n),
    run: (keys) => wake.sort(keys),
    reference: jsSortF32,
  });
}

export function benchSortGpu(wake: Wake, sizes: number[], trials: number): Promise<BenchResult[]> {
  return benchGpu<"f32">(wake, sizes, trials, {
    label: "wake:sort",
    bytesPerElement: GPU_BYTES_PER_ELEMENT,
    makeInput: (n) => randomArray("f32", n),
    run: (keys) => wake.sort(keys),
  });
}

export function benchSortCpu(sizes: number[], trials: number): BenchResult[] {
  return benchCpu<"f32">(sizes, trials, {
    label: "js:sort",
    makeInput: (n) => randomArray("f32", n),
    run: jsSortF32,
  });
}
