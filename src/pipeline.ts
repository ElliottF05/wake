const pipelineCaches = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

export function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  code: string,
): GPUComputePipeline {
  let cache = pipelineCaches.get(device);
  if (!cache) {
    cache = new Map();
    pipelineCaches.set(device, cache);
  }
  let pipeline = cache.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({
      label: `wake:${key}:shader`,
      code,
    });
    pipeline = device.createComputePipeline({
      label: `wake:${key}:pipeline`,
      layout: "auto",
      compute: { module },
    });
    cache.set(key, pipeline);
  }
  return pipeline;
}
