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
      f32: "0x1.fffffep+127f", // f32 max
      u32: "0xffffffffu",
      i32: "2147483647i",
    },
  },
  max: {
    combine: "max(a, b)",
    identity: {
      f32: "-0x1.fffffep+127f", // f32 min
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
  const workgroupSize = 256; // TODO: tune this and also maybe make it a passable param
  const dtype = wgslDtypeOf(input);
  const label = (what: string) => `wake:reduce:${what} (op=${op}, dtype=${dtype})`;
  const key = `reduce:${op}:${dtype}:${workgroupSize}`;
  const pipeline = await wake.getOrCreatePipeline(key, () => ({
    code: buildReduceShader(dtype, REDUCE_OPS[op], workgroupSize),
    shaderLabel: label("shader"),
    pipelineLabel: label("pipeline"),
  }));
  const p = Math.ceil(Math.sqrt(input.length));
  const scratch = wake.empty(p, input.view);
  const output = wake.empty(1, input.view);
  const bindGroup1 = wake.device.createBindGroup({
    label: label("bindGroup1"),
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.raw } },
      { binding: 1, resource: { buffer: scratch.raw } },
    ],
  });
  const bindGroup2 = wake.device.createBindGroup({
    label: label("bindGroup2"),
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: scratch.raw } },
      { binding: 1, resource: { buffer: output.raw } },
    ],
  });
  const encoder = wake.device.createCommandEncoder({ label: label("encoder") });
  const computePass = encoder.beginComputePass({ label: label("computePass") });
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup1);
  computePass.dispatchWorkgroups(Math.ceil(p / workgroupSize));
  computePass.setBindGroup(0, bindGroup2);
  computePass.dispatchWorkgroups(1);
  computePass.end();
  wake.device.queue.submit([encoder.finish()]);
  scratch.destroy();
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
function buildReduceShader(dtype: WgslDtype, op: ReduceOpWgsl, workgroupSize: number): string {
  const combine = op.combine;
  const identity = op.identity[dtype];
  return `
@group(0) @binding(0) var<storage, read> input: array<${dtype}>;
@group(0) @binding(1) var<storage, read_write> output: array<${dtype}>;

fn combine(a: ${dtype}, b: ${dtype}) -> ${dtype} {
  return ${combine};
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n_input = arrayLength(&input);
  let n_output = arrayLength(&output);
  if (gid.x >= n_output) {
    return;
  }
  var acc = ${identity};
  for (var i = gid.x; i < n_input; i += n_output) {
    acc = combine(acc, input[i]);
  }
  output[gid.x] = acc;
}
`;
}
