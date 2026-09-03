import type { Wake } from "./wake";

export type WakeSupportedArray = Float32Array | Uint32Array | Int32Array;

export type WakeArrayCtor<T extends WakeSupportedArray> = {
  new (buffer: ArrayBufferLike): T;
  readonly BYTES_PER_ELEMENT: number;
};

const BUFFER_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class WakeBuffer<T extends WakeSupportedArray = Float32Array> {
  private readonly wake: Wake;
  private readonly owned: boolean;
  private destroyed = false;

  readonly raw: GPUBuffer;
  readonly view: WakeArrayCtor<T>;
  readonly length: number;
  readonly byteLength: number;

  /**
   * Low-level constructor, this shouldn't be used directly. Use {@link Wake.buffer} to create
   * a buffer from a typed JS array, or {@link Wake.wrap} to wrap an existing GPUBuffer.
   */
  constructor(wake: Wake, raw: GPUBuffer, view: WakeArrayCtor<T>, owned = false) {
    if ((raw.usage & BUFFER_USAGE) !== BUFFER_USAGE) {
      throw new Error("WakeBuffer requires a GPUBuffer with STORAGE, COPY_SRC, and COPY_DST usage");
    }
    this.wake = wake;
    this.owned = owned;
    this.raw = raw;
    this.view = view;
    this.byteLength = raw.size;
    this.length = raw.size / view.BYTES_PER_ELEMENT;
  }

  async read(): Promise<T> {
    if (this.destroyed) {
      throw new Error("attempted to read destroyed WakeBuffer");
    }
    const device = this.wake.device;
    const staging = device.createBuffer({
      label: "wake:readback",
      size: this.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
      const encoder = device.createCommandEncoder({ label: "wake:read" });
      encoder.copyBufferToBuffer(this.raw, 0, staging, 0, this.byteLength);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const copy = staging.getMappedRange().slice(0);
      staging.unmap();
      return new this.view(copy);
    } finally {
      staging.destroy();
    }
  }

  write(data: T): void {
    if (this.destroyed) {
      throw new Error("attempted to write to a destroyed WakeBuffer");
    }
    if (data.byteLength !== this.byteLength) {
      throw new Error(
        `mismatched byte length when writing to a WakeBuffer, expected ${this.byteLength} bytes but got ${data.byteLength}`,
      );
    }
    this.wake.device.queue.writeBuffer(this.raw, 0, data);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.owned) this.raw.destroy();
  }
}

export function fromArray<T extends WakeSupportedArray>(wake: Wake, data: T): WakeBuffer<T> {
  if (
    !(data instanceof Float32Array || data instanceof Uint32Array || data instanceof Int32Array)
  ) {
    throw new Error("WakeBuffer expects a Float32Array, Uint32Array, or Int32Array");
  }
  const raw = wake.device.createBuffer({
    label: "wake:storage",
    size: data.byteLength,
    usage: BUFFER_USAGE,
    mappedAtCreation: true,
  });
  const view = data.constructor as WakeArrayCtor<T>;
  new view(raw.getMappedRange()).set(data);
  raw.unmap();
  return new WakeBuffer<T>(wake, raw, view, true);
}
