import { type WakeArray, WakeBuffer, type WakeDtype } from "./buffer";
import { sort } from "./radixSort";
import { reduce, type WakeReduceOp } from "./reduce";
import { encodeScan, scan } from "./scan";
import { readLastTimingMs } from "./timing";

/** The entry point to WAKE, wrapping the GPU device that its primitives run on. */
export class Wake {
  /** The GPU device that WAKE runs on. */
  readonly device: GPUDevice;
  private readonly owned: boolean;

  private constructor(device: GPUDevice, owned: boolean) {
    this.device = device;
    this.owned = owned;
  }

  /**
   * Creates a `Wake` instance on this machine's GPU. Throws if WebGPU is unavailable or the GPU
   * lacks required features (`subgroups` and `timestamp-query`). An existing GPU device can also
   * be passed in, for example to share it with other WebGPU code.
   */
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

  /**
   * Releases the GPU resources used by this `Wake` instance. If you passed your own GPU device to
   * `create()`, it is left for you to destroy.
   */
  destroy(): void {
    if (this.owned) this.device.destroy();
  }

  /**
   * Copies a `Float32Array` to a new f32 GPU buffer. Later changes to the array do not affect it.
   */
  buffer(data: Float32Array): WakeBuffer<"f32">;
  /**
   * Copies a `Uint32Array` to a new u32 GPU buffer. Later changes to the array do not affect it.
   */
  buffer(data: Uint32Array): WakeBuffer<"u32">;
  /** Copies a typed array to a new GPU buffer. Later changes to the array do not affect it. */
  buffer(data: WakeArray): WakeBuffer<WakeDtype>;
  buffer(data: WakeArray): WakeBuffer<WakeDtype> {
    return WakeBuffer.from(this, data);
  }

  /**
   * Wraps an existing `GPUBuffer` so WAKE can run on it without copying. Meant for apps that
   * already use WebGPU directly. The buffer needs STORAGE, COPY_SRC, and COPY_DST usage, and stays
   * owned by the caller.
   */
  wrap<D extends WakeDtype>(raw: GPUBuffer, dtype: D): WakeBuffer<D> {
    return WakeBuffer.wrap(this, raw, dtype);
  }

  /** Creates a new zero-filled GPU buffer with the given length and dtype. */
  empty<D extends WakeDtype>(length: number, dtype: D): WakeBuffer<D> {
    return WakeBuffer.empty(this, length, dtype);
  }

  /**
   * Combines all elements with the given op, returning a new one-element buffer containing the
   * result. `min` and `max` throw on an empty buffer.
   */
  reduce<D extends WakeDtype>(input: WakeBuffer<D>, op: WakeReduceOp): WakeBuffer<D> {
    return reduce(this, input, op);
  }

  /**
   * Sums all elements, returning a new one-element buffer containing the result. u32 sums wrap on
   * overflow.
   */
  sum<D extends WakeDtype>(input: WakeBuffer<D>): WakeBuffer<D> {
    return reduce(this, input, "sum");
  }

  /**
   * Finds the smallest element, returning a new one-element buffer containing the result. Throws
   * on an empty buffer.
   */
  min<D extends WakeDtype>(input: WakeBuffer<D>): WakeBuffer<D> {
    return reduce(this, input, "min");
  }

  /**
   * Finds the largest element, returning a new one-element buffer containing the result. Throws on
   * an empty buffer.
   */
  max<D extends WakeDtype>(input: WakeBuffer<D>): WakeBuffer<D> {
    return reduce(this, input, "max");
  }

  /**
   * Computes an exclusive prefix sum in place, returning the same buffer. Only u32 is supported,
   * and sums wrap on overflow.
   */
  scan(input: WakeBuffer<"u32">): WakeBuffer<"u32"> {
    return scan(this, input);
  }

  /**
   * Low-level version of `scan()` that records the scan into an existing compute pass instead of
   * running it immediately. Meant for building custom GPU pipelines that use scan as one step, as
   * WAKE's sort does. `scratch` needs STORAGE usage and at least `scanScratchBytes(input.length)`
   * bytes, and must be newly created or last used with an input of the same length.
   */
  encodeScan(pass: GPUComputePassEncoder, input: WakeBuffer<"u32">, scratch: GPUBuffer): void {
    encodeScan(this, pass, input, scratch);
  }

  /** Sorts in ascending order, in place, and returns the same buffer. The sort is stable. */
  sort<D extends WakeDtype>(keys: WakeBuffer<D>): WakeBuffer<D> {
    return sort(this, keys);
  }

  /**
   * Returns the GPU execution time of the most recent primitive, in milliseconds. This is slow and
   * is intended for benchmarking.
   */
  getLastTimingMs(): Promise<number> {
    return readLastTimingMs(this.device);
  }
}
