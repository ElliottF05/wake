import type { WakeBuffer, WakeSupportedArray } from "./buffer";
import { wgslDtypeOf } from "./ops";
import { scan } from "./scan";
import { MAX_NUM_WORKGROUPS, type Wake } from "./wake";
import { WakeLabel } from "./wakeLabel";

// These are all intentionally interdependent, don't change them without careful
// consideration.
const BITS_PER_PASS = 8;
const RADIX = 1 << BITS_PER_PASS; // 256
const WORKGROUP_SIZE = RADIX; // 256
// NUM_PASSES = 32 / BITS_PER_PASS (4) — add back with the per-pass loop

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
  const histogram = wake.empty(numWorkgroups * RADIX, Uint32Array);
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

  // Phase 1: Count
  computePass.setPipeline(count1Pipeline);
  computePass.setBindGroup(0, bindGroup1);
  computePass.dispatchWorkgroups(numWorkgroups);
  computePass.end();
  wake.device.queue.submit([encoder.finish()]);

  // Phase 2: Scan (prefix sum)
  const offsets = await scan(wake, histogram, "sum", false);
  histogram.destroy();

  // TODO: Phase 3 (reorder)
  offsets.destroy();
  return keys;
}

function buildCountShader(shift: number): string {
  return `
const RADIX = ${RADIX}u;
const DIGIT_MASK = RADIX - 1u;
const WORKGROUP_SIZE = ${WORKGROUP_SIZE}u;
const SHIFT = ${shift}u;

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32>;

var<workgroup> local_hist: array<atomic<u32>, RADIX>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(num_workgroups) num_wg: vec3u) {
  let workgroup_idx = wid.x;
  let num_workgroups = num_wg.x;
  let bin = lid.x;
  let n = arrayLength(&keys);

  // 1. zero the shared histogram
  atomicStore(&local_hist[bin], 0u);
  workgroupBarrier();

  // 2. coalesced grid-stride sweep of this section
  let section_size = (n + num_workgroups - 1u) / num_workgroups;
  let start = workgroup_idx * section_size;
  let end = min(start + section_size, n);
  for (var i = start + lid.x; i < end; i += WORKGROUP_SIZE) {
    let d = (keys[i] >> SHIFT) & DIGIT_MASK;
    atomicAdd(&local_hist[d], 1u);
  }
  workgroupBarrier();

  // 3. write this section's column into the global table
  histogram[bin * num_workgroups + workgroup_idx] = atomicLoad(&local_hist[bin]);
}
`;
}
