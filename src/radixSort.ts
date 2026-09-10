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
const NUM_PASSES = 32 / BITS_PER_PASS; // 4

// TODO: add optional `values` param?
export async function radixSort<T extends WakeSupportedArray>(
  wake: Wake,
  keys: WakeBuffer<T>,
): Promise<WakeBuffer<T>> {
  if (keys.length === 0) {
    return keys;
  }
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
  // TODO: this is weird syntax
  const passPipelines = await Promise.all(
    Array.from({ length: NUM_PASSES }, (_, pass) => {
      const shift = pass * BITS_PER_PASS;
      return Promise.all([
        pipelineFor(`count${pass}`, () => buildCountShader(shift)),
        pipelineFor(`reorder${pass}`, () => buildReorderShader(shift)),
      ]);
    }),
  );

  const n = keys.length;
  const numWorkgroups = Math.min(Math.ceil(n / WORKGROUP_SIZE), MAX_NUM_WORKGROUPS);
  const histogram = wake.empty(numWorkgroups * RADIX, Uint32Array);
  const scratch = wake.empty(n, Uint32Array);
  let src = keys.raw;
  let dst = scratch.raw;
  for (let pass = 0; pass < NUM_PASSES; pass++) {
    const [countPipeline, reorderPipeline] = passPipelines[pass];
    const passLabel = label.with("pass", `${pass}`);

    // Phase 1: Count
    const countBindGroup = wake.device.createBindGroup({
      label: passLabel.with("phase", "count").with("resource", "bindGroup").toString(),
      layout: countPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: histogram.raw } },
      ],
    });
    const countEncoder = wake.device.createCommandEncoder({
      label: passLabel.with("phase", "count").with("resource", "encoder").toString(),
    });
    const countPass = countEncoder.beginComputePass({
      label: passLabel.with("phase", "count").with("resource", "computePass").toString(),
    });
    countPass.setPipeline(countPipeline);
    countPass.setBindGroup(0, countBindGroup);
    countPass.dispatchWorkgroups(numWorkgroups);
    countPass.end();
    wake.device.queue.submit([countEncoder.finish()]);

    // Phase 2: Scan (prefix sum)
    // TODO: expose some sort of internal scan function that takes in the pass encoder etc
    const offsets = await scan(wake, histogram, "sum", false);

    // Phase 3: Reorder
    const reorderBindGroup = wake.device.createBindGroup({
      label: passLabel.with("phase", "reorder").with("resource", "bindGroup").toString(),
      layout: reorderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: offsets.raw } },
        { binding: 2, resource: { buffer: dst } },
      ],
    });
    const reorderEncoder = wake.device.createCommandEncoder({
      label: passLabel.with("phase", "reorder").with("resource", "encoder").toString(),
    });
    const reorderPass = reorderEncoder.beginComputePass({
      label: passLabel.with("phase", "reorder").with("resource", "computePass").toString(),
    });
    reorderPass.setPipeline(reorderPipeline);
    reorderPass.setBindGroup(0, reorderBindGroup);
    reorderPass.dispatchWorkgroups(numWorkgroups);
    reorderPass.end();
    wake.device.queue.submit([reorderEncoder.finish()]);

    offsets.destroy();
    [src, dst] = [dst, src];
  }

  histogram.destroy();
  scratch.destroy();
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

// TODO: this is just a one thread linear scan per workgroup, make it faster
function buildReorderShader(shift: number): string {
  return `
const RADIX = ${RADIX}u;
const DIGIT_MASK = RADIX - 1u;
const SHIFT = ${shift}u;

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3u,
        @builtin(num_workgroups) num_wg: vec3u) {
  let section_idx = wid.x;
  let num_sections = num_wg.x;
  let n = arrayLength(&keys_in);

  let section_size = (n + num_sections - 1u) / num_sections;
  let start = section_idx * section_size;
  let end = min(start + section_size, n);

  var cursor: array<u32, RADIX>;
  for (var d = 0u; d < RADIX; d += 1u) {
    cursor[d] = offsets[d * num_sections + section_idx];
  }

  for (var i = start; i < end; i += 1u) {
    let key = keys_in[i];
    let d = (key >> SHIFT) & DIGIT_MASK;
    keys_out[cursor[d]] = key;
    cursor[d] += 1u;
  }
}
`;
}
