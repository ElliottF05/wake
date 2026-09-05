import type { WakeBuffer, WakeSupportedArray } from "./buffer";
import {
  ASSOCIATIVE_OPS,
  type AssociativeOp,
  type AssociativeOpWgsl,
  type WgslDtype,
  wgslDtypeOf,
} from "./ops";
import type { Wake } from "./wake";

export async function scan<T extends WakeSupportedArray>(
  wake: Wake,
  input: WakeBuffer<T>,
  op: AssociativeOp,
  inclusive: boolean,
): Promise<WakeBuffer<T>> {
  const workgroupSize = 256; // TODO: tune this and also maybe make it a passable param
  const dtype = wgslDtypeOf(input);
  const label = (what: string) => `wake:scan:${what} (op=${op}, dtype=${dtype})`;
  const pipelineFor = (phase: string, buildShader: typeof buildPhase1Shader) =>
    wake.getOrCreatePipeline(`scan:${phase}:${op}:${dtype}:${workgroupSize}:${inclusive}`, () => ({
      code: buildShader(dtype, ASSOCIATIVE_OPS[op], workgroupSize, inclusive),
      shaderLabel: label(`shader:${phase}`),
      pipelineLabel: label(`pipeline:${phase}`),
    }));
  const [phase1Pipeline, phase2Pipeline, phase3Pipeline] = await Promise.all([
    pipelineFor("phase1", buildPhase1Shader),
    pipelineFor("phase2", buildPhase2Shader),
    pipelineFor("phase3", buildPhase3Shader),
  ]);
  const n = input.length;
  const p = Math.ceil(Math.sqrt(2 * n));
  const scratch = wake.empty(p, input.view);
  const output = wake.empty(n, input.view);
  const bindGroup1 = wake.device.createBindGroup({
    label: label("bindGroup1"),
    layout: phase1Pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.raw } },
      { binding: 1, resource: { buffer: scratch.raw } },
    ],
  });
  const bindGroup2 = wake.device.createBindGroup({
    label: label("bindGroup2"),
    layout: phase2Pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: scratch.raw } }],
  });
  const bindGroup3 = wake.device.createBindGroup({
    label: label("bindGroup3"),
    layout: phase3Pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.raw } },
      { binding: 1, resource: { buffer: scratch.raw } },
      { binding: 2, resource: { buffer: output.raw } },
    ],
  });
  const encoder = wake.device.createCommandEncoder({ label: label("encoder") });
  const computePass = encoder.beginComputePass({ label: label("computePass") });
  computePass.setPipeline(phase1Pipeline);
  computePass.setBindGroup(0, bindGroup1);
  computePass.dispatchWorkgroups(Math.ceil(p / workgroupSize));
  computePass.setPipeline(phase2Pipeline);
  computePass.setBindGroup(0, bindGroup2);
  computePass.dispatchWorkgroups(1);
  computePass.setPipeline(phase3Pipeline);
  computePass.setBindGroup(0, bindGroup3);
  computePass.dispatchWorkgroups(Math.ceil(p / workgroupSize));
  computePass.end();
  wake.device.queue.submit([encoder.finish()]);
  scratch.destroy();
  return output;
}

// TODO: make these shaders faster based on data from running on 3080

function buildPhase1Shader(
  dtype: WgslDtype,
  op: AssociativeOpWgsl,
  workgroupSize: number,
  _inclusive: boolean,
): string {
  const combine = op.combine;
  const identity = op.identity[dtype];
  return `
@group(0) @binding(0) var<storage, read> input: array<${dtype}>;
@group(0) @binding(1) var<storage, read_write> scratch: array<${dtype}>;

fn combine(a: ${dtype}, b: ${dtype}) -> ${dtype} {
  return ${combine};
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = arrayLength(&input);
  let p = arrayLength(&scratch);
  if (gid.x >= p) {
    return;
  }
  let block_size = (n + p - 1u) / p;
  let start = gid.x * block_size;
  let end = min(start + block_size, n);
  var acc = ${identity};
  for (var i = start; i < end; i++) {
    acc = combine(acc, input[i]);
  }
  scratch[gid.x] = acc;
}
`;
}

function buildPhase2Shader(
  dtype: WgslDtype,
  op: AssociativeOpWgsl,
  _workgroupSize: number,
  _inclusive: boolean,
): string {
  const combine = op.combine;
  const identity = op.identity[dtype];
  return `
@group(0) @binding(0) var<storage, read_write> scratch: array<${dtype}>;

fn combine(a: ${dtype}, b: ${dtype}) -> ${dtype} {
  return ${combine};
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = arrayLength(&scratch);
  if (gid.x != 0) {
    return;
  }
  var prev = ${identity};
  for (var i = 0u; i < p; i++) {
    let x = scratch[i];
    scratch[i] = prev;
    prev = combine(prev, x);
  }
}
`;
}

function buildPhase3Shader(
  dtype: WgslDtype,
  op: AssociativeOpWgsl,
  workgroupSize: number,
  inclusive: boolean,
): string {
  const combine = op.combine;
  const step = inclusive
    ? `prev = combine(prev, input[i]);\n    output[i] = prev;`
    : `output[i] = prev;\n    prev = combine(prev, input[i]);`;
  return `
@group(0) @binding(0) var<storage, read> input: array<${dtype}>;
@group(0) @binding(1) var<storage, read> scratch: array<${dtype}>;
@group(0) @binding(2) var<storage, read_write> output: array<${dtype}>;

fn combine(a: ${dtype}, b: ${dtype}) -> ${dtype} {
  return ${combine};
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = arrayLength(&input);
  let p = arrayLength(&scratch);
  let b = gid.x;
  if (b >= p) {
    return;
  }
  let block_size = (n + p - 1u) / p;
  let start = b * block_size;
  let end = min(start + block_size, n);
  var prev = scratch[b];
  for (var i = start; i < end; i++) {
    ${step}
  }
}
`;
}
