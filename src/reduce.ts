import type { WakeBuffer, WakeSupportedArray } from "./buffer";
import {
  ASSOCIATIVE_OPS,
  type AssociativeOp,
  type AssociativeOpWgsl,
  type WgslDtype,
  wgslCombineFn,
  wgslDtypeOf,
} from "./ops";
import type { Wake } from "./wake";
import { WakeLabel } from "./wakeLabel";

export async function reduce<T extends WakeSupportedArray>(
  wake: Wake,
  input: WakeBuffer<T>,
  op: AssociativeOp,
): Promise<WakeBuffer<T>> {
  const workgroupSize = 256; // TODO: tune this and also maybe make it a passable param
  const dtype = wgslDtypeOf(input);
  const label = new WakeLabel().with("primitive", "reduce").with("op", op).with("dtype", dtype);
  const key = `reduce:${op}:${dtype}:${workgroupSize}`;
  const pipeline = await wake.getOrCreatePipeline(key, () => ({
    code: buildReduceShader(dtype, ASSOCIATIVE_OPS[op], workgroupSize),
    label: label,
  }));
  const p = Math.ceil(Math.sqrt(input.length));
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
  computePass.dispatchWorkgroups(Math.ceil(p / workgroupSize));
  computePass.setBindGroup(0, bindGroup2);
  computePass.dispatchWorkgroups(1);
  computePass.end();
  wake.device.queue.submit([encoder.finish()]);
  scratch.destroy();
  return output;
}

// TODO: make this faster based on data from running on 3080
function buildReduceShader(dtype: WgslDtype, op: AssociativeOpWgsl, workgroupSize: number): string {
  const identity = op.identity[dtype];
  return `
@group(0) @binding(0) var<storage, read> input: array<${dtype}>;
@group(0) @binding(1) var<storage, read_write> output: array<${dtype}>;

${wgslCombineFn(dtype, op)}

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
