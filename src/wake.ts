// TODO: add comments to public-facing contents (not just in this file) when things are more stable

import { type WakeArray, WakeBuffer, type WakeDtype } from "./buffer";
import { sort } from "./radixSort";
import { reduce, type WakeReduceOp } from "./reduce";
import { encodeScan, scan } from "./scan";
import { readLastTimingMs } from "./timing";

export class Wake {
  readonly device: GPUDevice;
  private readonly owned: boolean;

  private constructor(device: GPUDevice, owned: boolean) {
    this.device = device;
    this.owned = owned;
  }

  static async create(device?: GPUDevice): Promise<Wake> {
    const owned = device === undefined;
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
    return new Wake(device, owned);
  }

  destroy(): void {
    if (this.owned) this.device.destroy();
  }

  buffer(data: Float32Array): WakeBuffer<"f32">;
  buffer(data: Uint32Array): WakeBuffer<"u32">;
  buffer(data: WakeArray): WakeBuffer<WakeDtype>;
  buffer(data: WakeArray): WakeBuffer<WakeDtype> {
    return WakeBuffer.from(this, data);
  }

  wrap<D extends WakeDtype>(raw: GPUBuffer, dtype: D): WakeBuffer<D> {
    return WakeBuffer.wrap(this, raw, dtype);
  }

  empty<D extends WakeDtype>(length: number, dtype: D): WakeBuffer<D> {
    return WakeBuffer.empty(this, length, dtype);
  }

  reduce<D extends WakeDtype>(input: WakeBuffer<D>, op: WakeReduceOp): WakeBuffer<D> {
    return reduce(this, input, op);
  }

  sum<D extends WakeDtype>(input: WakeBuffer<D>): WakeBuffer<D> {
    return reduce(this, input, "sum");
  }

  min<D extends WakeDtype>(input: WakeBuffer<D>): WakeBuffer<D> {
    return reduce(this, input, "min");
  }

  max<D extends WakeDtype>(input: WakeBuffer<D>): WakeBuffer<D> {
    return reduce(this, input, "max");
  }

  scan(input: WakeBuffer<"u32">): WakeBuffer<"u32"> {
    return scan(this, input);
  }

  encodeScan(pass: GPUComputePassEncoder, input: WakeBuffer<"u32">, scratch: GPUBuffer): void {
    encodeScan(this, pass, input, scratch);
  }

  sort<D extends WakeDtype>(keys: WakeBuffer<D>): WakeBuffer<D> {
    return sort(this, keys);
  }

  getLastTimingMs(): Promise<number> {
    return readLastTimingMs(this.device);
  }
}
