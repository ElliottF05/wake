// TODO: add comments to public-facing contents (not just in this file) when things are more stable

import {
  fromArray,
  type WakeArray,
  type WakeArrayCtor,
  type WakeBuffer,
  wrapBuffer,
} from "./buffer";

export interface WakeOptions {
  device?: GPUDevice;
  powerPreference?: GPUPowerPreference;
}

export class Wake {
  readonly device: GPUDevice;

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
      throw new Error("No WebGPU adapter available");
    }
    try {
      const device = await adapter.requestDevice({ label: "wake:device" });
      return new Wake(device);
    } catch (cause) {
      throw new Error("Failed to acquire a WebGPU device", { cause });
    }
  }

  buffer<T extends WakeArray>(data: T): WakeBuffer<T> {
    return fromArray(this, data);
  }

  wrap<T extends WakeArray>(raw: GPUBuffer, view: WakeArrayCtor<T>): WakeBuffer<T> {
    return wrapBuffer(this, raw, view);
  }

  destroy(): void {
    this.device.destroy();
  }
}
