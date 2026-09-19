// TODO: add comments to public-facing contents (not just in this file) when things are more stable
// TODO: add `wake.sum()`, `wake.min()`, etc for convenience? Also maybe `wakeBuffer.sum()` etc

import {
  emptyBuffer,
  fromArray,
  WakeBuffer,
  type WakeSupportedArray,
  type WakeSupportedArrayConstructor,
} from "./buffer";
import type { AssociativeOp } from "./ops";
import { reduce } from "./reduce";
import { scan } from "./scan";
import type { WakeLabel } from "./wakeLabel";

export const MAX_NUM_WORKGROUPS = 1024; // TODO: tune this, and in the future potentially have some runtime measurement of what is best

export interface WakeOptions {
  device?: GPUDevice;
  powerPreference?: GPUPowerPreference;
}

export class Wake {
  readonly device: GPUDevice;
  private readonly pipelineCache = new Map<string, Promise<GPUComputePipeline>>();

  private constructor(device: GPUDevice) {
    this.device = device;
  }

  static isSupported(): boolean {
    return globalThis.navigator?.gpu != null;
  }

  static async create(options: WakeOptions = {}): Promise<Wake> {
    const { device, powerPreference = "high-performance" } = options;
    if (device) {
      return new Wake(device);
    }
    if (!Wake.isSupported()) {
      throw new Error("WebGPU is not supported in this environment");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) {
      throw new Error("no WebGPU adapter available");
    }
    try {
      const device = await adapter.requestDevice({
        label: "wake:device",
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
        },
      });
      return new Wake(device);
    } catch (cause) {
      throw new Error("failed to acquire a WebGPU device", { cause });
    }
  }

  destroy(): void {
    this.device.destroy();
  }

  buffer<T extends WakeSupportedArray>(data: T): WakeBuffer<T> {
    return fromArray(this, data);
  }

  wrap<T extends WakeSupportedArray>(
    raw: GPUBuffer,
    view: WakeSupportedArrayConstructor<T>,
  ): WakeBuffer<T> {
    return new WakeBuffer(this, raw, view);
  }

  empty<T extends WakeSupportedArray>(
    length: number,
    view: WakeSupportedArrayConstructor<T>,
  ): WakeBuffer<T> {
    return emptyBuffer(this, length, view);
  }

  reduce<T extends WakeSupportedArray>(
    input: WakeBuffer<T>,
    op: AssociativeOp,
  ): Promise<WakeBuffer<T>> {
    return reduce(this, input, op);
  }

  scan<T extends WakeSupportedArray>(
    input: WakeBuffer<T>,
    op: AssociativeOp,
    inclusive: boolean,
  ): Promise<WakeBuffer<T>> {
    return scan(this, input, op, inclusive);
  }

  getOrCreatePipeline(
    key: string,
    build: () => {
      code: string;
      label: WakeLabel;
    },
  ): Promise<GPUComputePipeline> {
    let cached = this.pipelineCache.get(key);
    if (!cached) {
      const { code, label } = build();
      const module = this.device.createShaderModule({
        label: label.with("resource", "shader").toString(),
        code,
      });
      cached = this.device.createComputePipelineAsync({
        label: label.with("resource", "pipeline").toString(),
        layout: "auto",
        compute: { module },
      });
      this.pipelineCache.set(key, cached);
    }
    return cached;
  }
}
