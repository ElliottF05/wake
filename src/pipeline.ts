import type { WakeLabel } from "./wakeLabel";

const pipelineCaches = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline>>>();

export function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  build: () => {
    code: string;
    label: WakeLabel;
  },
): Promise<GPUComputePipeline> {
  let cache = pipelineCaches.get(device);
  if (!cache) {
    cache = new Map();
    pipelineCaches.set(device, cache);
  }
  let cached = cache.get(key);
  if (!cached) {
    const { code, label } = build();
    const module = device.createShaderModule({
      label: label.with("resource", "shader").toString(),
      code,
    });
    cached = device.createComputePipelineAsync({
      label: label.with("resource", "pipeline").toString(),
      layout: "auto",
      compute: { module },
    });
    cache.set(key, cached);
  }
  return cached;
}
