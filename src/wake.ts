// TODO: add comments to public-facing contents (not just in this file) when things are more stable

import {
  emptyBuffer,
  fromArray,
  WakeBuffer,
  type WakeSupportedArray,
  type WakeSupportedArrayConstructor,
} from "./buffer";
import { reduce } from "./reduce";

export interface WakeOptions {
  device?: GPUDevice;
  powerPreference?: GPUPowerPreference;
}

export class Wake {
  readonly device: GPUDevice;
  private readonly pipelineCache = new Map();

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

  sum<T extends WakeSupportedArray>(input: WakeBuffer<T>): Promise<WakeBuffer<T>> {
    return reduce(this, input, "sum", this.pipelineCache);
  }

  min<T extends WakeSupportedArray>(input: WakeBuffer<T>): Promise<WakeBuffer<T>> {
    return reduce(this, input, "min", this.pipelineCache);
  }

  max<T extends WakeSupportedArray>(input: WakeBuffer<T>): Promise<WakeBuffer<T>> {
    return reduce(this, input, "max", this.pipelineCache);
  }
}
