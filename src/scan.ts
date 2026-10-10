import type { WakeBuffer } from "./buffer";
import { getOrCreatePipeline } from "./pipeline";
import { timestampWrites } from "./timing";
import type { Wake } from "./wake";

const SUBGROUP_SIZE = 32; // MUST NOT BE CHANGED
const WORKGROUP_SIZE = 256; // must be a multiple of SUBGROUP_SIZE (32)
const ELEMS_PER_THREAD = 32;

const SUBGROUPS_PER_WORKGROUP = WORKGROUP_SIZE / SUBGROUP_SIZE;
const ELEMS_PER_BLOCK = SUBGROUP_SIZE * ELEMS_PER_THREAD;

type ScratchLevel = {
  byteOffset: number;
  byteSize: number;
  parentLength: number;
};

export function scanScratchBytes(n: number): number {
  const levels = calculateScratchLevels(n);
  if (levels.length === 0) {
    return 0;
  }
  const last = levels[levels.length - 1];
  return last.byteOffset + last.byteSize;
}

export function scan(wake: Wake, input: WakeBuffer<"u32">): WakeBuffer<"u32"> {
  const encoder = wake.device.createCommandEncoder({ label: "wake:scan:encoder" });
  const pass = encoder.beginComputePass({
    label: "wake:scan:pass",
    timestampWrites: timestampWrites(wake.device),
  });
  const scratch = wake.device.createBuffer({
    size: scanScratchBytes(input.length),
    usage: GPUBufferUsage.STORAGE,
  });
  encodeScan(wake, pass, input, scratch);
  pass.end();
  wake.device.queue.submit([encoder.finish()]);
  scratch.destroy();
  return input;
}

export function encodeScan(
  wake: Wake,
  pass: GPUComputePassEncoder,
  input: WakeBuffer<"u32">,
  scratch: GPUBuffer,
): void {
  if (input.length === 0) {
    return;
  }

  const phase1Pipeline = getOrCreatePipeline(wake.device, "scan:phase1", phase1Code);
  const phase2Pipeline = getOrCreatePipeline(wake.device, "scan:phase2", phase2Code);

  const levels = calculateScratchLevels(input.length);

  const encodeLevel = (pipeline: GPUComputePipeline, phase: string, i: number): void => {
    const level = levels[i];
    const parent: GPUBufferBinding =
      i === 0
        ? { buffer: input.raw }
        : { buffer: scratch, offset: levels[i - 1].byteOffset, size: levels[i - 1].byteSize };
    const child: GPUBufferBinding = {
      buffer: scratch,
      offset: level.byteOffset,
      size: level.byteSize,
    };

    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      wake.device.createBindGroup({
        label: `wake:scan:${phase}:level${i}:bindGroup`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: parent },
          { binding: 1, resource: child },
        ],
      }),
    );
    pass.dispatchWorkgroups(calculateNumWorkgroups(wake, level.parentLength));
  };

  for (let i = 0; i < levels.length - 1; i++) {
    encodeLevel(phase1Pipeline, "phase1", i);
  }
  for (let i = levels.length - 1; i >= 0; i--) {
    encodeLevel(phase2Pipeline, "phase2", i);
  }
}

function calculateScratchLevels(n: number): ScratchLevel[] {
  if (n === 1) {
    return [{ byteOffset: 0, byteSize: 4, parentLength: 1 }];
  }

  const ALIGNMENT = 256;
  const levels: ScratchLevel[] = [];
  let byteOffset = 0;

  while (n > 1) {
    const parentLength = n;
    n = Math.ceil(n / ELEMS_PER_BLOCK);
    const byteSize = 4 * n;
    levels.push({ byteOffset, byteSize, parentLength });
    byteOffset += byteSize;
    byteOffset = Math.ceil(byteOffset / ALIGNMENT) * ALIGNMENT;
  }

  return levels;
}

function calculateNumWorkgroups(wake: Wake, n: number): number {
  const targetWorkgroups = Math.ceil(n / ELEMS_PER_BLOCK);
  const maxWorkgroups = wake.device.limits.maxComputeWorkgroupsPerDimension;
  return Math.min(targetWorkgroups, maxWorkgroups);
}

const phase1Code = `
enable subgroups;

const SUBGROUP_SIZE: u32 = ${SUBGROUP_SIZE};
const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE};
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD};
const SUBGROUPS_PER_WORKGROUP: u32 = ${SUBGROUPS_PER_WORKGROUP};
const ELEMS_PER_BLOCK: u32 = ${ELEMS_PER_BLOCK};

@group(0u) @binding(0u) var<storage, read_write> data: array<u32>;
@group(0u) @binding(1u) var<storage, read_write> block_sums: array<u32>;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn phase1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_invocation_id) sid: u32,
) {
    let data_len = arrayLength(&data);
    let num_blocks = div_ceil(data_len, ELEMS_PER_BLOCK);
    let num_sg = num_wg.x * SUBGROUPS_PER_WORKGROUP;
    let blocks_per_sg = div_ceil(num_blocks, num_sg);
    let sg = lid.x / SUBGROUP_SIZE;
    let global_sg = wid.x * SUBGROUPS_PER_WORKGROUP + sg;
    let base_block = global_sg * blocks_per_sg;

    // multiple blocks may be assigned in the case where there is a workgroup 
    // invocation limit
    for (var b = 0u; b < blocks_per_sg; b++) {
        let curr_block = base_block + b;

        // thread-local reduction over ELEMS_PER_THREAD elements, strided for
        // coalesced addressing
        var sum = 0u;
        for (var k = 0u; k < ELEMS_PER_THREAD; k++) {
            let idx = curr_block * ELEMS_PER_BLOCK + sid + k * SUBGROUP_SIZE;
            if (idx < data_len) {
                sum += data[idx];
            }
        }

        // subgroup reduction
        sum = subgroupAdd(sum);

        // write block total to block_sums
        if (sid == 0u && curr_block < num_blocks) {
            block_sums[curr_block] = sum;
        }
    }
}`;

const phase2Code = `
enable subgroups;

const SUBGROUP_SIZE: u32 = ${SUBGROUP_SIZE};
const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE};
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD};
const SUBGROUPS_PER_WORKGROUP: u32 = ${SUBGROUPS_PER_WORKGROUP};
const ELEMS_PER_BLOCK: u32 = ${ELEMS_PER_BLOCK};

@group(0u) @binding(0u) var<storage, read_write> data: array<u32>;
@group(0u) @binding(1u) var<storage, read_write> block_sums: array<u32>;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn phase2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_invocation_id) sid: u32,
) {
    let data_len = arrayLength(&data);
    let num_blocks = div_ceil(data_len, ELEMS_PER_BLOCK);
    let num_sg = num_wg.x * SUBGROUPS_PER_WORKGROUP;
    let blocks_per_sg = div_ceil(num_blocks, num_sg);
    let sg = lid.x / SUBGROUP_SIZE;
    let global_sg = wid.x * SUBGROUPS_PER_WORKGROUP + sg;
    let base_block = global_sg * blocks_per_sg;

    for (var b = 0u; b < blocks_per_sg; b++) {
        let curr_block = base_block + b;

        // incr begins as the previous block sum
        var incr = 0u;
        if (curr_block < num_blocks) {
            incr = block_sums[curr_block];
        }

        // scan each portion of SUBGROUP_SIZE elements, incrementing the running incr
        for (var k = 0u; k < ELEMS_PER_THREAD; k++) {
            let data_idx = curr_block * ELEMS_PER_BLOCK + sid + k * SUBGROUP_SIZE;
            var x = 0u;
            if (data_idx < data_len) {
                x = data[data_idx];
            }

            // subgroup scan
            let incl = subgroupInclusiveAdd(x);
            if (data_idx < data_len) {
                data[data_idx] = incr + incl - x;
            }
            incr += subgroupBroadcast(incl, SUBGROUP_SIZE - 1u);
        }
    }
}`;
