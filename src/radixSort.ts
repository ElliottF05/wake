import type { WakeBuffer, WakeDtype } from "./buffer";
import { getOrCreatePipeline } from "./pipeline";
import { encodeScan, scanScratchBytes } from "./scan";
import { timestampWrites } from "./timing";
import type { Wake } from "./wake";

const ELEMS_PER_THREAD = 8;

const SUBGROUP_SIZE = 32; // MUST NOT BE CHANGED
const RADIX = 256; // don't change this either
const WORKGROUP_SIZE = RADIX;

const ELEMS_PER_BLOCK = WORKGROUP_SIZE * ELEMS_PER_THREAD;

export function sort<D extends WakeDtype>(wake: Wake, keys: WakeBuffer<D>): WakeBuffer<D> {
  const n = keys.length;
  if (n <= 1) {
    return keys;
  }

  const encoder = wake.device.createCommandEncoder({ label: "wake:sort:encoder" });
  const pass = encoder.beginComputePass({
    label: "wake:sort:pass",
    timestampWrites: timestampWrites(wake.device),
  });

  const keysScratch = wake.device.createBuffer({
    label: "wake:sort:keysScratch",
    size: keys.byteLength,
    usage: GPUBufferUsage.STORAGE,
  });
  const keyBuffers = [keys.raw, keysScratch];

  const numBlocks = Math.ceil(n / ELEMS_PER_BLOCK);
  const hist = wake.empty(RADIX * numBlocks, "u32");

  const scanScratch = wake.device.createBuffer({
    label: "wake:sort:scanScratch",
    size: scanScratchBytes(hist.length),
    usage: GPUBufferUsage.STORAGE,
  });

  const pipelines = [0, 8, 16, 24].map((radixOffset) => {
    const countPipeline = getOrCreatePipeline(
      wake.device,
      `sort:count:offset${radixOffset}`,
      buildCountCode(radixOffset),
    );
    const reorderPipeline = getOrCreatePipeline(
      wake.device,
      `sort:reorder:offset${radixOffset}`,
      buildReorderCode(radixOffset),
    );
    return { countPipeline, reorderPipeline };
  });

  const numWorkgroups = Math.min(numBlocks, wake.device.limits.maxComputeWorkgroupsPerDimension);

  pipelines.forEach(({ countPipeline, reorderPipeline }, passIndex) => {
    pass.setPipeline(countPipeline);
    const countBindGroup = wake.device.createBindGroup({
      label: `wake:sort:count:pass${passIndex}:bindGroup`,
      layout: countPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: keyBuffers[passIndex % 2] } },
        { binding: 1, resource: { buffer: hist.raw } },
      ],
    });
    pass.setBindGroup(0, countBindGroup);
    pass.dispatchWorkgroups(numWorkgroups);

    encodeScan(wake, pass, hist, scanScratch);

    pass.setPipeline(reorderPipeline);
    const reorderBindGroup = wake.device.createBindGroup({
      label: `wake:sort:reorder:pass${passIndex}:bindGroup`,
      layout: reorderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: keyBuffers[passIndex % 2] } },
        { binding: 1, resource: { buffer: hist.raw } },
        { binding: 2, resource: { buffer: keyBuffers[(passIndex + 1) % 2] } },
      ],
    });
    pass.setBindGroup(0, reorderBindGroup);
    pass.dispatchWorkgroups(numWorkgroups);
  });

  pass.end();
  wake.device.queue.submit([encoder.finish()]);
  keysScratch.destroy();
  hist.destroy();
  scanScratch.destroy();
  return keys;
}

function buildCountCode(radix_offset: number): string {
  return `
const RADIX: u32 = ${RADIX};
const WORKGROUP_SIZE: u32 = RADIX;
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD};
const ELEMS_PER_BLOCK: u32 = WORKGROUP_SIZE * ELEMS_PER_THREAD;

const RADIX_OFFSET: u32 = ${radix_offset};

@group(0u) @binding(0u) var<storage, read> keys_in: array<u32>;
@group(0u) @binding(1u) var<storage, read_write> hist: array<u32>;

var<workgroup> local_hist: array<atomic<u32>, RADIX>;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn count(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let n = arrayLength(&keys_in);
    let num_blocks = div_ceil(n, ELEMS_PER_BLOCK);
    let blocks_per_wg = div_ceil(num_blocks, num_wg.x);
    let base_block = wid.x * blocks_per_wg;

    atomicStore(&local_hist[lid.x], 0u);
    workgroupBarrier();

    for (var b = 0u; b < blocks_per_wg; b++) {
        let curr_block = base_block + b;
        if (curr_block >= num_blocks) {
            break;
        }

        for (var k = 0u; k < ELEMS_PER_THREAD; k++) {
            let idx = curr_block * ELEMS_PER_BLOCK + lid.x + k * WORKGROUP_SIZE;
            if (idx < n) {
                let digit = (keys_in[idx] >> RADIX_OFFSET) & (RADIX - 1u);
                atomicAdd(&local_hist[digit], 1u);
            }
        }
        workgroupBarrier();

        hist[lid.x * num_blocks + curr_block] = atomicExchange(&local_hist[lid.x], 0u);
        workgroupBarrier();
    }
}
`;
}

function buildReorderCode(radix_offset: number): string {
  return `
enable subgroups;

const RADIX: u32 = ${RADIX};
const WORKGROUP_SIZE: u32 = RADIX;
const SUBGROUP_SIZE: u32 = ${SUBGROUP_SIZE};
const NUM_SUBGROUPS: u32 = WORKGROUP_SIZE / SUBGROUP_SIZE;
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD};
const ELEMS_PER_BLOCK: u32 = WORKGROUP_SIZE * ELEMS_PER_THREAD;
const ELEMS_PER_SG: u32 = ELEMS_PER_BLOCK / NUM_SUBGROUPS;

const RADIX_OFFSET: u32 = ${radix_offset};

@group(0u) @binding(0u) var<storage, read> keys_in: array<u32>;
@group(0u) @binding(1u) var<storage, read> hist: array<u32>;
@group(0u) @binding(2u) var<storage, read_write> keys_out: array<u32>;

var<workgroup> sg_hist: array<array<u32, RADIX>, NUM_SUBGROUPS>;

fn digit_of(x: u32) -> u32 {
    return (x >> RADIX_OFFSET) & (RADIX - 1u);
}

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reorder(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_invocation_id) sid: u32,
) {
    let n = arrayLength(&keys_in);
    let num_blocks = div_ceil(n, ELEMS_PER_BLOCK);
    let blocks_per_wg = div_ceil(num_blocks, num_wg.x);
    let base_block = wid.x * blocks_per_wg;
    let sg = lid.x / SUBGROUP_SIZE;

    const INVALID_DIGIT: u32 = 0xFFFFFFFFu;

    for (var b = 0u; b < blocks_per_wg; b++) {
        let curr_block = base_block + b;
        if (curr_block >= num_blocks) {
            break;
        }

        for (var s = 0u; s < NUM_SUBGROUPS; s++) {
            sg_hist[s][lid.x] = 0u;
        }
        workgroupBarrier();

        var partial_rank: array<u32, ELEMS_PER_THREAD>;
        var local_keys: array<u32, ELEMS_PER_THREAD>;

        let base = curr_block * ELEMS_PER_BLOCK + sg * ELEMS_PER_SG + sid;
        for (var k = 0u; k < ELEMS_PER_THREAD; k++) {
            let idx = base + k * SUBGROUP_SIZE;
            var key = 0u;
            var digit = INVALID_DIGIT;
            if (idx < n) {
                key = keys_in[idx];
                local_keys[k] = key;
                digit = digit_of(key);
            }

            var peers = ~subgroupBallot(digit == INVALID_DIGIT).x;
            for (var bit = 0u; bit < 8u; bit++) {
                let bit_set = (digit >> bit) & 1u;
                // bit_set == 1 -> mask 0 (keep ballot)
                // bit_set == 0 -> mask 0xFFFFFFFF (complement)
                peers &= subgroupBallot(bit_set == 1u).x ^ (bit_set - 1u);
            }

            let leader = countTrailingZeros(peers);
            let num_peers = countOneBits(peers);
            var prev = 0u;
            if (sid == leader && digit != INVALID_DIGIT) {
                prev = sg_hist[sg][digit];
                sg_hist[sg][digit] += num_peers;
            }
            prev = subgroupShuffle(prev, leader);

            let num_prev_lanes_with_digit = countOneBits(peers & ((1u << sid) - 1u));
            partial_rank[k] = prev + num_prev_lanes_with_digit;
        }
        workgroupBarrier();

        var sum = 0u;
        for (var s = 0u; s < NUM_SUBGROUPS; s++) {
            let x = sg_hist[s][lid.x];
            sg_hist[s][lid.x] = sum;
            sum += x;
        }
        workgroupBarrier();

        for (var k = 0u; k < ELEMS_PER_THREAD; k++) {
            if (base + k * SUBGROUP_SIZE >= n) {
                break;
            }
            let key = local_keys[k];
            let digit = digit_of(key);
            let idx = hist[digit * num_blocks + curr_block] + sg_hist[sg][digit] + partial_rank[k];
            keys_out[idx] = key;
        }

        workgroupBarrier();
    }
}
`;
}
