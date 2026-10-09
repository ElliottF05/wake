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
  let cached = cache.get(key);
  if (!cached) {
    const module = device.createShaderModule({
      label: `wake:${key}:shader`,
      code,
    });
    cached = device.createComputePipeline({
      label: `wake:${key}:pipeline`,
      layout: "auto",
      compute: { module },
    });
    cache.set(key, cached);
  }
  return cached;
}
