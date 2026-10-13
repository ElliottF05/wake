const TIMESTAMP_COUNT = 2;
const TIMESTAMP_BYTES = TIMESTAMP_COUNT * 8;

const querySets = new WeakMap<GPUDevice, GPUQuerySet>();

function getOrCreateQuerySet(device: GPUDevice): GPUQuerySet {
  let querySet = querySets.get(device);
  if (!querySet) {
    querySet = device.createQuerySet({
      label: "wake:timing:querySet",
      type: "timestamp",
      count: TIMESTAMP_COUNT,
    });
    querySets.set(device, querySet);
  }
  return querySet;
}

export function timestampWrites(device: GPUDevice): GPUComputePassTimestampWrites {
  return {
    querySet: getOrCreateQuerySet(device),
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  };
}

export async function readLastTimingMs(device: GPUDevice): Promise<number> {
  const querySet = getOrCreateQuerySet(device);
  const resolved = device.createBuffer({
    label: "wake:timing:resolved",
    size: TIMESTAMP_BYTES,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const staging = device.createBuffer({
    label: "wake:timing:staging",
    size: TIMESTAMP_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: "wake:timing:encoder" });
    encoder.resolveQuerySet(querySet, 0, TIMESTAMP_COUNT, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, staging, 0, TIMESTAMP_BYTES);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(staging.getMappedRange());
    const elapsedNs = timestamps[1] - timestamps[0];
    staging.unmap();
    return Number(elapsedNs) / 1e6;
  } finally {
    resolved.destroy();
    staging.destroy();
  }
}
