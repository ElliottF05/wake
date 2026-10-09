import type { WakeBuffer } from "./buffer";
import type { Wake } from "./wake";

const SUBGROUP_SIZE = 32; // MUST NOT BE CHANGED
const WORKGROUP_SIZE = 256; // must be a multiple of SUBGROUP_SIZE (32)
const ELEMS_PER_THREAD = 32;

const ELEMS_PER_BLOCK = SUBGROUP_SIZE * ELEMS_PER_THREAD;

export function scanScratchLength(n: number): number {
  throw new Error("not implemented");
}

export function scan(wake: Wake, input: WakeBuffer<"u32">): WakeBuffer<"u32"> {
  throw new Error("not implemented");
}

export function encodeScan(
  wake: Wake,
  pass: GPUComputePassEncoder,
  input: WakeBuffer<"u32">,
  scratch: WakeBuffer<"u32">,
): void {
  throw new Error("not implemented");
}

// TODO: handle edge cases where n = 0 or n = 1?
function scratchOffsets(n: number): number[][] {
  const reductionFactor = SUBGROUP_SIZE * ELEMS_PER_THREAD;
  const alignment = 256;

  const offsets = [];
  let offset = 0;

  while (n > 1) {
    n = Math.ceil(n / reductionFactor);
    const numBytes = 4 * n;
    offsets.push([offset, numBytes]);
    offset += numBytes;
    offset = Math.ceil(offset / alignment) * alignment;
  }

  return offsets;
}

function numWorkgroups(wake: Wake, n: number): number {
  const targetWorkgroups = Math.ceil(n / ELEMS_PER_BLOCK);
  const maxWorkgroups = wake.device.limits.maxComputeWorkgroupsPerDimension;
  return Math.min(targetWorkgroups, maxWorkgroups);
}

// TODO: should i use the js value of ELEMS_PER_BLOCK?
const code = `
const SUBGROUP_SIZE: u32 = ${SUBGROUP_SIZE};
const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE};
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD};
const SUBGROUPS_PER_WORKGROUP: u32 = WORKGROUP_SIZE / SUBGROUP_SIZE;
const ELEMS_PER_BLOCK: u32 = SG_SIZE * ELEMS_PER_THREAD;

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
        if (curr_block >= num_blocks) {
            break;
        }

        // thread-local reduction over ELEMS_PER_THREAD elements, strided for
        // coalesced addressing
        var sum = 0u;
        for (var k = 0u; k < ELEMS_PER_THREAD; k++) {
            let idx = curr_block * ELEMS_PER_BLOCK + sid + k * SG_SIZE;
            if (idx < data_len) {
                sum += data[idx];
            }
        }

        // subgroup reduction
        sum = subgroupAdd(sum);

        // write block total to block_sums
        if (sid == 0u) {
            block_sums[curr_block] = sum;
        }
    }
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
    let num_sg = num_wg.x * SUBGROUPS_PER_WORKGROUP
    let blocks_per_sg = div_ceil(num_blocks, num_sg);
    let sg = lid.x / SUBGROUP_SIZE;
    let global_sg = wid.x * SUBGROUPS_PER_WORKGROUP + sg;
    let base_block = global_sg * blocks_per_sg;

    for (var b = 0u; b < blocks_per_sg; b++) {
        let curr_block = base_block + b;
        if (curr_block >= num_blocks) {
            break;
        }

        // incr begins as the previous block sum
        var incr = block_sums[curr_block];

        // scan each portion of SG_SIZE elements, incrementing the running incr
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
