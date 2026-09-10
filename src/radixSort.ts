import type { WakeBuffer, WakeSupportedArray } from "./buffer";
import { wgslDtypeOf } from "./ops";
import { MAX_NUM_WORKGROUPS, type Wake } from "./wake";
import { WakeLabel } from "./wakeLabel";

// These 3 are intentionally interdependent, don't change them without careful 
// consideration.
const WORKGROUP_SIZE = 256;
const BITS_PER_PASS = 8; // 2^8 = 256
const RADIX = 256;

// TODO: add optional `values` param?
export async function radixSort<T extends WakeSupportedArray>(
  wake: Wake,
  keys: WakeBuffer<T>,
): Promise<WakeBuffer<T>> {
  const dtype = wgslDtypeOf(keys);
  if (dtype !== "u32") {
    throw Error("only u32 is supported by radixSort for now"); // TODO: fix this
  }
  const label = new WakeLabel().with("primitive", "radixSort").with("dtype", dtype);
  const pipelineFor = (phase: string, buildCode: () => string) =>
    wake.getOrCreatePipeline(`radixSort:${phase}`, () => ({
      code: buildCode(),
      label: label.with("phase", phase),
    }));
  // TODO: add more here
  const [count1Pipeline] = await Promise.all([pipelineFor("count1", () => buildCountShader(0))]);
  const n = keys.length;
  const numWorkgroups = Math.min(Math.ceil(n / WORKGROUP_SIZE), MAX_NUM_WORKGROUPS);
  const histogram = wake.empty(numWorkgroups * WORKGROUP_SIZE, Uint32Array);
  const bindGroup1 = wake.device.createBindGroup({
    label: label.with("phase", "count1").with("resource", "bindGroup").toString(),
    layout: count1Pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: keys.raw } },
      { binding: 1, resource: { buffer: histogram.raw } },
    ],
  });
  const encoder = wake.device.createCommandEncoder({
    label: label.with("resource", "encoder").toString(),
  });
  const computePass = encoder.beginComputePass({
    label: label.with("resource", "computePass").toString(),
  });
  computePass.setPipeline(count1Pipeline);
  computePass.setBindGroup(0, bindGroup1);
  computePass.dispatchWorkgroups(numWorkgroups);
  computePass.end();
  wake.device.queue.submit([encoder.finish()]);
  return keys;
}

function buildCountShader(shift: number): string {
  return `
const RADIX = ${RADIX}u;
const MASK = ${RADIX - 1}u;
const SHIFT = ${shift};

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32>;

var<workgroup> local_hist: array<atomic<u32>, RADIX>;

@compute @workgroup_size(RADIX)
fn main(@builtin(workgroup_id) wid: vec3u,
        @builtin(local_invocation_id) lid: vec3u) {
  let m = wid.x; // index of current workgroup
  let n = arrayLength(&keys);
  let p = arrayLength(&histogram) / RADIX;

  // 1. zero the shared histogram
  atomicStore(&local_hist[lid.x], 0u);
  workgroupBarrier();

  // 2. coalesced grid-stride sweep of this section
  let section_size = (n + p - 1u) / p;
  let start = m * section_size;
  let end = min(start + section_size, n);
  for (var i = start + lid.x; i < end; i += ${WORKGROUP_SIZE}) {
    let d = (keys[i] >> SHIFT) & MASK;
    atomicAdd(&local_hist[d], 1u);
  }
  workgroupBarrier();

  // 3. write this section's column into the global table
  histogram[lid.x * p + m] = atomicLoad(&local_hist[lid.x]);
}
`;
}
