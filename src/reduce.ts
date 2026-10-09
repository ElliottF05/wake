import type { WakeBuffer, WakeSupportedArray } from "./buffer";
import {
  ASSOCIATIVE_OPS,
  type AssociativeOp,
  type AssociativeOpWgsl,
  type WgslDtype,
  wgslCombineFn,
  wgslDtypeOf,
} from "./ops";
import { getOrCreatePipeline } from "./pipeline";
import type { Wake } from "./wake";
import { WakeLabel } from "./wakeLabel";

const WORKGROUP_SIZE = 256;

export async function reduce<T extends WakeSupportedArray>(
  wake: Wake,
  input: WakeBuffer<T>,
  op: AssociativeOp,
): Promise<WakeBuffer<T>> {
  if (input.length === 0) {
    if (op !== "sum") {
      throw new Error(`reduce: cannot ${op} an empty buffer`);
    }
    return wake.empty(1, input.view);
  }
  const dtype = wgslDtypeOf(input);
  const label = new WakeLabel().with("primitive", "reduce").with("op", op).with("dtype", dtype);
  const key = `reduce:${op}:${dtype}`;
  const pipeline = await getOrCreatePipeline(wake.device, key, () => ({
    code: buildReduceShader(dtype, ASSOCIATIVE_OPS[op]),
    label: label,
  }));
  const n = input.length;
  const p = Math.ceil(Math.sqrt(n));
  const numWorkgroups = Math.ceil(p / WORKGROUP_SIZE);
  const scratch = wake.empty(p, input.view);
  const output = wake.empty(1, input.view);
  const bindGroup1 = wake.device.createBindGroup({
    label: label.with("resource", "bindGroup1").toString(),
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.raw } },
      { binding: 1, resource: { buffer: scratch.raw } },
    ],
  });
  const bindGroup2 = wake.device.createBindGroup({
    label: label.with("resource", "bindGroup2").toString(),
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: scratch.raw } },
      { binding: 1, resource: { buffer: output.raw } },
    ],
  });
  const encoder = wake.device.createCommandEncoder({
    label: label.with("resource", "encoder").toString(),
  });
  const computePass = encoder.beginComputePass({
    label: label.with("resource", "computePass").toString(),
  });
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup1);
  computePass.dispatchWorkgroups(numWorkgroups);
  computePass.setBindGroup(0, bindGroup2);
  computePass.dispatchWorkgroups(1);
  computePass.end();
  wake.device.queue.submit([encoder.finish()]);
  scratch.destroy();
  return output;
}

// TODO: make this faster based on data from running on 3080
function buildReduceShader(dtype: WgslDtype, op: AssociativeOpWgsl): string {
  const identity = op.identity[dtype];
  // TODO: extract consts out to the top?
  return `
@group(0) @binding(0) var<storage, read> input: array<${dtype}>;
@group(0) @binding(1) var<storage, read_write> output: array<${dtype}>;

${wgslCombineFn(dtype, op)}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = arrayLength(&input);
  let p = arrayLength(&output);
  if (gid.x >= p) {
    return;
  }
  var acc = ${identity};
  for (var i = gid.x; i < n; i += p) {
    acc = combine(acc, input[i]);
  }
  output[gid.x] = acc;
}
`;
}
