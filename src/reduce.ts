import type { WakeBuffer, WakeDtype } from "./buffer";
import { getOrCreatePipeline } from "./pipeline";
import { timestampWrites } from "./timing";
import type { Wake } from "./wake";

const SUBGROUP_SIZE = 32; // MUST NOT BE CHANGED
const WORKGROUP_SIZE = 256; // must be a multiple of SUBGROUP_SIZE (32)
const ELEMS_PER_THREAD = 32;

const SUBGROUPS_PER_WORKGROUP = WORKGROUP_SIZE / SUBGROUP_SIZE;
const ELEMS_PER_BLOCK = SUBGROUP_SIZE * ELEMS_PER_THREAD;
const ELEMS_PER_WORKGROUP = WORKGROUP_SIZE * ELEMS_PER_THREAD;

export type WakeReduceOp = "sum" | "min" | "max";

export function reduce<D extends WakeDtype>(
  wake: Wake,
  input: WakeBuffer<D>,
  op: WakeReduceOp,
): WakeBuffer<D> {
  const n = input.length;
  if (n === 0 && op !== "sum") {
    throw new Error(`cannot take ${op} of an empty input`);
  }
  const output = wake.empty(1, input.dtype);
  if (n === 0) {
    return output;
  }

  const scratch = [
    Math.ceil(n / ELEMS_PER_BLOCK),
    Math.ceil(Math.ceil(n / ELEMS_PER_BLOCK) / ELEMS_PER_BLOCK),
  ].map((length, i) =>
    wake.device.createBuffer({
      label: `wake:reduce:scratch${i}`,
      size: 4 * length,
      usage: GPUBufferUsage.STORAGE,
    }),
  );

  const encoder = wake.device.createCommandEncoder({ label: "wake:reduce:encoder" });
  const pass = encoder.beginComputePass({
    label: "wake:reduce:pass",
    timestampWrites: timestampWrites(wake.device),
  });

  const pipeline = getOrCreatePipeline(
    wake.device,
    `reduce:${op}:${input.dtype}`,
    buildShaderCode(op, input.dtype),
  );
  pass.setPipeline(pipeline);

  let length = n;
  let level = 0;
  do {
    const isLast = length <= ELEMS_PER_BLOCK;
    const src = level === 0 ? input.raw : scratch[(level - 1) % 2];
    const dst = isLast ? output.raw : scratch[level % 2];

    const bindGroup = wake.device.createBindGroup({
      label: `wake:reduce:level${level}:bindGroup`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
      ],
    });
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(calculateNumWorkgroups(wake, length));

    length = Math.ceil(length / ELEMS_PER_BLOCK);
    level++;
  } while (length > 1);

  pass.end();
  wake.device.queue.submit([encoder.finish()]);
  for (const buffer of scratch) {
    buffer.destroy();
  }
  return output;
}

function calculateNumWorkgroups(wake: Wake, n: number): number {
  const targetWorkgroups = Math.ceil(n / ELEMS_PER_WORKGROUP);
  const maxWorkgroups = wake.device.limits.maxComputeWorkgroupsPerDimension;
  return Math.min(targetWorkgroups, maxWorkgroups);
}

const REDUCE_OPS: Record<
  WakeReduceOp,
  {
    combine: string;
    subgroupReduce: string;
    identity: Record<WakeDtype, string>;
  }
> = {
  sum: {
    combine: "a + b",
    subgroupReduce: "subgroupAdd",
    identity: { u32: "0u", f32: "0.0f" },
  },
  min: {
    combine: "min(a, b)",
    subgroupReduce: "subgroupMin",
    identity: { u32: "0xffffffffu", f32: "0x1.fffffep+127f" }, // f32 max
  },
  max: {
    combine: "max(a, b)",
    subgroupReduce: "subgroupMax",
    identity: { u32: "0u", f32: "-0x1.fffffep+127f" }, // f32 min
  },
};

function buildShaderCode(op: WakeReduceOp, dtype: WakeDtype): string {
  const { combine, subgroupReduce, identity } = REDUCE_OPS[op];
  return `
enable subgroups;

const SUBGROUP_SIZE: u32 = ${SUBGROUP_SIZE};
const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE};
const ELEMS_PER_THREAD: u32 = ${ELEMS_PER_THREAD};
const SUBGROUPS_PER_WORKGROUP: u32 = ${SUBGROUPS_PER_WORKGROUP};
const ELEMS_PER_BLOCK: u32 = ${ELEMS_PER_BLOCK};

@group(0u) @binding(0u) var<storage, read_write> src: array<${dtype}>;
@group(0u) @binding(1u) var<storage, read_write> dst: array<${dtype}>;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

fn combine(a: ${dtype}, b: ${dtype}) -> ${dtype} {
    return ${combine};
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_invocation_id) sid: u32,
) {
    let src_len = arrayLength(&src);
    let num_blocks = div_ceil(src_len, ELEMS_PER_BLOCK);
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
        var acc = ${identity[dtype]};
        for (var k = 0u; k < ELEMS_PER_THREAD; k++) {
            let idx = curr_block * ELEMS_PER_BLOCK + sid + k * SUBGROUP_SIZE;
            if (idx < src_len) {
                acc = combine(acc, src[idx]);
            }
        }

        // subgroup reduction
        acc = ${subgroupReduce}(acc);

        // write this block's total to dst
        if (sid == 0u && curr_block < num_blocks) {
            dst[curr_block] = acc;
        }
    }
}`;
}
