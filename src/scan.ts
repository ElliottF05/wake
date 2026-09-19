import type { WakeBuffer, WakeSupportedArray } from "./buffer";
import {
  ASSOCIATIVE_OPS,
  type AssociativeOp,
  type AssociativeOpWgsl,
  type WgslDtype,
  wgslCombineFn,
  wgslDtypeOf,
} from "./ops";
import { MAX_NUM_WORKGROUPS, type Wake } from "./wake";
import { WakeLabel } from "./wakeLabel";

const WORKGROUP_SIZE = 256;

export async function scan<T extends WakeSupportedArray>(
  wake: Wake,
  input: WakeBuffer<T>,
  op: AssociativeOp,
): Promise<WakeBuffer<T>> {
  if (input.length === 0) {
    return wake.empty(0, input.view);
  }
  const K = 16;
  const dtype = wgslDtypeOf(input);
  const label = new WakeLabel().with("primitive", "scan").with("op", op).with("dtype", dtype);
  const pipelineFor = (phase: string, key: string, buildCode: () => string) =>
    wake.getOrCreatePipeline(`scan:${phase}:${key}`, () => ({
      code: buildCode(),
      label: label.with("phase", phase),
    }));
  const phase1Pipeline = await pipelineFor("phase1", `${op}:${dtype}`, () =>
    buildPhase1Shader(dtype, ASSOCIATIVE_OPS[op], K),
  );
  const phase2Pipeline = await pipelineFor("phase2", `${op}:${dtype}`, () =>
    buildPhase2Shader(dtype, ASSOCIATIVE_OPS[op]),
  );
  const phase3Pipeline = await pipelineFor("phase3", `${op}:${dtype}`, () =>
    buildPhase3Shader(dtype, ASSOCIATIVE_OPS[op], K),
  );
  const n = input.length;
  const numWorkgroups = Math.min(Math.ceil(n / (K * WORKGROUP_SIZE)), MAX_NUM_WORKGROUPS);
  const scratch = wake.empty(numWorkgroups, input.view);
  const output = wake.empty(n, input.view);
  const bindGroup1 = wake.device.createBindGroup({
    label: label.with("phase", "phase1").with("resource", "bindGroup").toString(),
    layout: phase1Pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.raw } },
      { binding: 1, resource: { buffer: scratch.raw } },
    ],
  });
  const bindGroup2 = wake.device.createBindGroup({
    label: label.with("phase", "phase2").with("resource", "bindGroup").toString(),
    layout: phase2Pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: scratch.raw } }],
  });
  const bindGroup3 = wake.device.createBindGroup({
    label: label.with("phase", "phase3").with("resource", "bindGroup").toString(),
    layout: phase3Pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.raw } },
      { binding: 1, resource: { buffer: scratch.raw } },
      { binding: 2, resource: { buffer: output.raw } },
    ],
  });
  const encoder = wake.device.createCommandEncoder({
    label: label.with("resource", "encoder").toString(),
  });
  const computePass = encoder.beginComputePass({
    label: label.with("resource", "computePass").toString(),
  });
  computePass.setPipeline(phase1Pipeline);
  computePass.setBindGroup(0, bindGroup1);
  computePass.dispatchWorkgroups(numWorkgroups);
  computePass.setPipeline(phase2Pipeline);
  computePass.setBindGroup(0, bindGroup2);
  computePass.dispatchWorkgroups(1);
  computePass.setPipeline(phase3Pipeline);
  computePass.setBindGroup(0, bindGroup3);
  computePass.dispatchWorkgroups(numWorkgroups);
  computePass.end();
  wake.device.queue.submit([encoder.finish()]);
  scratch.destroy();
  return output;
}

// TODO: make these shaders faster based on data from running on 3080
// TODO: extract consts out to the top of each shader?
// Builds `num_workgroups` partial reductions
function buildPhase1Shader(dtype: WgslDtype, op: AssociativeOpWgsl, K: number): string {
  return `
const WORKGROUP_SIZE = ${WORKGROUP_SIZE}u;
const K = ${K}u;
const IDENTITY = ${op.identity[dtype]};

@group(0) @binding(0) var<storage, read> input: array<${dtype}>;
@group(0) @binding(1) var<storage, read_write> scratch: array<${dtype}>;

var<workgroup> partials: array<${dtype}, WORKGROUP_SIZE>;

${wgslCombineFn(dtype, op)}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) num_workgroups: vec3u
) {
  let n = arrayLength(&input);
  let elems_per_tile = K * WORKGROUP_SIZE;
  let num_tiles = (n + elems_per_tile - 1u) / elems_per_tile;
  let tiles_per_wg = (num_tiles + num_workgroups.x - 1u) / num_workgroups.x;
  let elems_per_wg = tiles_per_wg * elems_per_tile;

  let start = wid.x * elems_per_wg;
  let end = min(start + elems_per_wg, n);
  var acc = IDENTITY;
  for (var i = start + lid.x; i < end; i += WORKGROUP_SIZE) {
    acc = combine(acc, input[i]);
  }
  partials[lid.x] = acc;

  workgroupBarrier();

  if (lid.x == 0) {
    acc = IDENTITY;
    for (var i = 0u; i < WORKGROUP_SIZE; i++) {
      acc = combine(acc, partials[i]);
    }
    scratch[wid.x] = acc;
  }
}
`;
}

// Linear scan over the `num_workgroups` partial reductions from the previous phase.
function buildPhase2Shader(dtype: WgslDtype, op: AssociativeOpWgsl): string {
  return `
const IDENTITY = ${op.identity[dtype]};

@group(0) @binding(0) var<storage, read_write> scratch: array<${dtype}>;

${wgslCombineFn(dtype, op)}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0) {
    return;
  }
  var prev = IDENTITY;
  for (var i = 0u; i < arrayLength(&scratch); i++) {
    let x = scratch[i];
    scratch[i] = prev;
    prev = combine(prev, x);
  }
}
`;
}

function buildPhase3Shader(dtype: WgslDtype, op: AssociativeOpWgsl, K: number): string {
  return `
const WORKGROUP_SIZE = ${WORKGROUP_SIZE}u;
const K = ${K}u;
const IDENTITY = ${op.identity[dtype]};

@group(0) @binding(0) var<storage, read> input: array<${dtype}>;
@group(0) @binding(1) var<storage, read> scratch: array<${dtype}>;
@group(0) @binding(2) var<storage, read_write> output: array<${dtype}>;

var<workgroup> tile: array<${dtype}, K * WORKGROUP_SIZE>;
var<workgroup> partials: array<${dtype}, WORKGROUP_SIZE>;
var<workgroup> tile_total: ${dtype};

${wgslCombineFn(dtype, op)}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) num_workgroups: vec3u
) {
  let n = arrayLength(&input);
  let elems_per_tile = K * WORKGROUP_SIZE;
  let num_tiles = (n + elems_per_tile - 1u) / elems_per_tile;
  let tiles_per_wg = (num_tiles + num_workgroups.x - 1u) / num_workgroups.x;
  let elems_per_wg = tiles_per_wg * elems_per_tile;

  var running = scratch[wid.x];
  for (var t = 0u; t < tiles_per_wg; t++) {

    for (var k = 0u; k < K; k++) {
      let i = lid.x + k * WORKGROUP_SIZE;
      let j = (wid.x * tiles_per_wg + t) * elems_per_tile + i;
      if (j < n) {
        tile[i] = input[j];
      } else {
        tile[i] = IDENTITY;
      }
    }

    workgroupBarrier();

    var acc = IDENTITY;
    for (var k = 0u; k < K; k++) {
      let i = lid.x * K + k;
      acc = combine(acc, tile[i]);
    }
    partials[lid.x] = acc;

    workgroupBarrier();

    if (lid.x == 0) {
      var prev = IDENTITY;
      for (var i = 0u; i < WORKGROUP_SIZE; i++) {
        let x = partials[i];
        partials[i] = prev;
        prev = combine(prev, x);
      }
      tile_total = prev;
    }

    workgroupBarrier();

    var prev = combine(running, partials[lid.x]);
    for (var k = 0u; k < K; k++) {
      let i = lid.x * K + k;
      let x = tile[i];
      tile[i] = prev;
      prev = combine(prev, x);
    }

    workgroupBarrier();

    for (var k = 0u; k < K; k++) {
      let i = lid.x + k * WORKGROUP_SIZE;
      let j = (wid.x * tiles_per_wg + t) * elems_per_tile + i;
      if (j < n) {
        output[j] = tile[i];
      }
    }

    running = combine(running, tile_total);

    workgroupBarrier();
  }
}
`;
}
