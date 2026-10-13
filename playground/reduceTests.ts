import type { Wake } from "../src/index";
import { reduce, type WakeReduceOp } from "../src/reduce";
import { testCorrectness } from "./harness";
import { randomArray } from "./inputs";

const OPS: WakeReduceOp[] = ["sum", "min", "max"];

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
  for (const op of OPS) {
    await testCorrectness<"u32">(wake, sizes, {
      label: `reduce:${op}:u32`,
      // min/max of an empty input throws, by design
      sizes: (n) => n > 0 || op === "sum",
      makeInput: (n) => randomArray("u32", n),
      toBuffer: (w, data) => w.buffer(data),
      run: (input) => wake.reduce(input, op),
      reference: (input) => jsReduceU32(input, op),
    });
  }

  for (const op of OPS) {
    await testCorrectness<"f32">(wake, sizes, {
      label: `reduce:${op}:f32`,
      sizes: (n) => n > 0 || op === "sum",
      makeInput: (n) => randomArray("f32", n),
      toBuffer: (w, data) => w.buffer(data),
      run: (input) => wake.reduce(input, op),
      reference: (input) => jsReduceF32(input, op),
      tolerance: op === "sum" ? f32SumTolerance : undefined,
    });
  }
}
