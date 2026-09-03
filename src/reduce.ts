import type { WakeBuffer, WakeSupportedArray } from "./buffer";
import type { Wake } from "./wake";

export type ReduceOp = "sum" | "min" | "max";

type WgslDtype = "f32" | "u32" | "i32";

type ReduceOpWgsl = {
  combine: string;
  identity: Record<WgslDtype, string>;
};

const REDUCE_OPS: Record<ReduceOp, ReduceOpWgsl> = {
  sum: {
    combine: "a + b",
    identity: { f32: "0f", u32: "0u", i32: "0i" },
  },
  min: {
    combine: "min(a, b)",
    identity: {
      f32: "bitcast<f32>(0x7f800000u)", // inf
      u32: "0xffffffffu",
      i32: "2147483647i",
    },
  },
  max: {
    combine: "max(a, b)",
    identity: {
      f32: "bitcast<f32>(0xff800000u)", // -inf
      u32: "0u",
      i32: "-2147483647i - 1i",
    },
  },
};

export async function reduce<T extends WakeSupportedArray>(
  wake: Wake,
  input: WakeBuffer<T>,
  op: ReduceOp,
): Promise<WakeBuffer<T>> {
  const dtype = wgslDtypeOf(input);
  const label = (what: string) => `wake:reduce:${what} (op=${op}, dtype=${dtype})`;
  const module = wake.device.createShaderModule({
    label: label("shader"),
    code: buildReduceShader(dtype, REDUCE_OPS[op]),
  });
  const pipeline = await wake.device.createComputePipelineAsync({
    label: label("pipeline"),
    layout: "auto",
    compute: { module },
  });
  const output = wake.buffer(new input.view(new ArrayBuffer(input.view.BYTES_PER_ELEMENT)));
  const bindGroup = wake.device.createBindGroup({
    label: label("bindGroup"),
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.raw } },
      { binding: 1, resource: { buffer: output.raw } },
    ],
  });
  const encoder = wake.device.createCommandEncoder({ label: label("encoder") });
  const computePass = encoder.beginComputePass({ label: label("computePass") });
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup);
  computePass.dispatchWorkgroups(1);
  computePass.end();
  wake.device.queue.submit([encoder.finish()]);
  return output;
}

function wgslDtypeOf(buffer: WakeBuffer<WakeSupportedArray>): WgslDtype {
  if (buffer.view === Float32Array) return "f32";
  if (buffer.view === Uint32Array) return "u32";
  if (buffer.view === Int32Array) return "i32";
  throw new Error(
    `unsupported element type ${buffer.view.name}, expected Float32Array, Uint32Array, or Int32Array`,
  );
}

// for now, this just does a linear scan on one thread, very inefficient!
// TODO: make this faster
function buildReduceShader(dtype: WgslDtype, op: ReduceOpWgsl): string {
  const combine = op.combine;
  const identity = op.identity[dtype];
  return `
@group(0) @binding(0) var<storage, read> input: array<${dtype}>;
@group(0) @binding(1) var<storage, read_write> output: array<${dtype}>;

fn combine(a: ${dtype}, b: ${dtype}) -> ${dtype} {
  return ${combine};
}

@compute @workgroup_size(1)
fn main() {
  var n = arrayLength(&input);
  var out = ${identity};
  for (var i: u32 = 0; i < n; i++) {
    out = combine(out, input[i]);
  }
  output[0] = out;
}
`;
}
