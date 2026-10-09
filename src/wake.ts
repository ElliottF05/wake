// TODO: add comments to public-facing contents (not just in this file) when things are more stable
// TODO: add `wake.sum()`, `wake.min()`, etc for convenience? Also maybe `wakeBuffer.sum()` etc

import { type WakeArray, WakeBuffer, type WakeDtype } from "./buffer";
import type { AssociativeOp } from "./ops";
import { reduce } from "./reduce";
import { encodeScan, scan } from "./scan";

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

  buffer(data: Float32Array): WakeBuffer<"f32">;
  buffer(data: Uint32Array): WakeBuffer<"u32">;
  buffer(data: WakeArray): WakeBuffer<WakeDtype> {
    return WakeBuffer.from(this, data);
  }

  wrap<D extends WakeDtype>(raw: GPUBuffer, dtype: D): WakeBuffer<D> {
    return WakeBuffer.wrap(this, raw, dtype);
  }

  empty<D extends WakeDtype>(length: number, dtype: D): WakeBuffer<D> {
    return WakeBuffer.empty(this, length, dtype);
  }

  // reduce<D extends WakeDtype>(input: WakeBuffer<D>, op: AssociativeOp): WakeBuffer<D> {
  //   return reduce(this, input, op);
  // }

  scan(input: WakeBuffer<"u32">): WakeBuffer<"u32"> {
    return scan(this, input);
  }

  encodeScan(
    pass: GPUComputePassEncoder,
    input: WakeBuffer<"u32">,
    scratch: WakeBuffer<"u32">,
  ): void {
    encodeScan(this, pass, input, scratch);
  }
}
