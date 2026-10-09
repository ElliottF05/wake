const pipelineCaches = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

export function getOrCreatePipeline(
  device: GPUDevice,
  shaderKey: string,
  code: string,
  entryPoint: string,
): GPUComputePipeline {
  let cache = pipelineCaches.get(device);
  if (!cache) {
    cache = new Map();
    pipelineCaches.set(device, cache);
  }
  const pipelineKey = `${shaderKey}:${entryPoint}`;
  let pipeline = cache.get(pipelineKey);
  if (!pipeline) {
    const module = device.createShaderModule({
      label: `wake:${pipelineKey}:shader`,
      code,
    });
    pipeline = device.createComputePipeline({
      label: `wake:${pipelineKey}:pipeline`,
      layout: "auto",
      compute: { module, entryPoint },
    });
    cache.set(pipelineKey, pipeline);
  }
  return pipeline;
}
