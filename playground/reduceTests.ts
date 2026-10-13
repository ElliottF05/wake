import type { Wake } from "../src/index";
import type { WakeReduceOp } from "../src/reduce";
import { type BenchResult, benchCpu, benchGpu, testCorrectness } from "./harness";
import { randomArray } from "./inputs";

const OPS: WakeReduceOp[] = ["sum", "min", "max"];

// one pass: reads n elements, writes a single value. the same on both sides, so GB/s is
// directly comparable between them
const BYTES_PER_ELEMENT = 4;

const F32_EPS = 2 ** -24;
const TOLERANCE_SLACK = 16;

function jsReduceU32(input: Uint32Array, op: WakeReduceOp): Uint32Array {
  const output = new Uint32Array(1);
  if (op === "sum") {
    let acc = 0;
    for (let i = 0; i < input.length; i++) {
      acc = (acc + input[i]) >>> 0;
    }
    output[0] = acc;
    return output;
  }

  let acc = op === "min" ? 0xffffffff : 0;
  for (let i = 0; i < input.length; i++) {
    acc = op === "min" ? Math.min(acc, input[i]) : Math.max(acc, input[i]);
  }
  output[0] = acc;
  return output;
}

function jsReduceF32(input: Float32Array, op: WakeReduceOp): Float32Array {
  const output = new Float32Array(1);
  if (op === "sum") {
    let acc = 0;
    for (let i = 0; i < input.length; i++) {
      acc += input[i];
    }
    output[0] = acc;
    return output;
  }

  let acc = op === "min" ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  for (let i = 0; i < input.length; i++) {
    acc = op === "min" ? Math.min(acc, input[i]) : Math.max(acc, input[i]);
  }
  output[0] = acc;
  return output;
}

function f32SumTolerance(input: Float32Array): number {
  let magnitude = 0;
  for (let i = 0; i < input.length; i++) {
    magnitude += Math.abs(input[i]);
  }
  return TOLERANCE_SLACK * F32_EPS * magnitude;
}

export async function testReduceCorrectness(wake: Wake, sizes: number[]): Promise<void> {
  const nonEmptySizes = sizes.filter((n) => n > 0);

  for (const op of OPS) {
    await testCorrectness<"u32">(wake, op === "sum" ? sizes : nonEmptySizes, {
      label: `reduce:${op}:u32`,
      makeInput: (n) => randomArray("u32", n),
      run: (input) => wake.reduce(input, op),
      reference: (input) => jsReduceU32(input, op),
    });
  }

  for (const op of OPS) {
    await testCorrectness<"f32">(wake, op === "sum" ? sizes : nonEmptySizes, {
      label: `reduce:${op}:f32`,
      makeInput: (n) => randomArray("f32", n),
      run: (input) => wake.reduce(input, op),
      reference: (input) => jsReduceF32(input, op),
      tolerance: op === "sum" ? f32SumTolerance : undefined,
    });
  }
}

export function benchReduceGpu(
  wake: Wake,
  sizes: number[],
  trials: number,
): Promise<BenchResult[]> {
  return benchGpu<"f32">(wake, sizes, trials, {
    label: "wake:reduce",
    bytesPerElement: BYTES_PER_ELEMENT,
    makeInput: (n) => randomArray("f32", n),
    run: (input) => wake.reduce(input, "sum"),
  });
}

export function benchReduceCpu(sizes: number[], trials: number): BenchResult[] {
  return benchCpu<"f32">(sizes, trials, {
    label: "js:reduce",
    bytesPerElement: BYTES_PER_ELEMENT,
    makeInput: (n) => randomArray("f32", n),
    run: (input) => jsReduceF32(input, "sum"),
  });
}
