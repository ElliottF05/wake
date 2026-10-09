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

export class Wake {
  readonly device: GPUDevice;

  private constructor(device: GPUDevice) {
    this.device = device;
  }

  static async create(device?: GPUDevice): Promise<Wake> {
    const requiredFeatures: GPUFeatureName[] = ["subgroups", "timestamp-query"];
    if (!device) {
      if (globalThis.navigator?.gpu == null) {
        throw new Error("WebGPU is not supported in this environment");
      }
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) {
        throw new Error("no WebGPU adapter available");
      }
      device = await adapter.requestDevice({
        label: "wake:device",
        requiredFeatures,
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
          maxBufferSize: adapter.limits.maxBufferSize,
        },
      });
    } else {
      for (const feature of requiredFeatures) {
        if (!device.features.has(feature)) {
          throw new Error(`wake requires the "${feature}" WebGPU feature, which this device lacks`);
        }
      }
    }
    return new Wake(device);
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
  ): Promise<WakeBuffer<T>> {
    return scan(this, input, op);
  }
}
