import type { Wake } from "./wake";

export type WakeArray = Float32Array | Uint32Array | Int32Array;

export type WakeArrayCtor<T extends WakeArray> = {
  new (buffer: ArrayBufferLike): T;
  readonly BYTES_PER_ELEMENT: number;
};

export class WakeBuffer<T extends WakeArray = Float32Array> {
  private readonly wake: Wake;
  private readonly owned: boolean;
  private destroyed = false;

  readonly raw: GPUBuffer;
  readonly view: WakeArrayCtor<T>;
  readonly length: number;
  readonly byteLength: number;

  /**
   * Low-level constructor, this shouldn't be called directly. Use {@link Wake.buffer} to create a buffer from a
   * typed array, or {@link Wake.wrap} to wrap an existing GPUBuffer.
   */
  constructor(wake: Wake, raw: GPUBuffer, view: WakeArrayCtor<T>, owned = false) {
    this.wake = wake;
    this.owned = owned;
    this.raw = raw;
    this.view = view;
    this.byteLength = raw.size;
    this.length = raw.size / view.BYTES_PER_ELEMENT;
  }

  async read(): Promise<T> {
    if (this.destroyed) {
      throw new Error("WakeBuffer.read called on a destroyed buffer");
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
      throw new Error("WakeBuffer.write called on a destroyed buffer");
    }
    if (data.byteLength !== this.byteLength) {
      throw new Error(
        `WakeBuffer.write expected ${this.byteLength} bytes but received ${data.byteLength}`,
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

export function fromArray<T extends WakeArray>(wake: Wake, data: T): WakeBuffer<T> {
  if (
    !(data instanceof Float32Array || data instanceof Uint32Array || data instanceof Int32Array)
  ) {
    throw new Error("wake.buffer expects a Float32Array, Uint32Array, or Int32Array");
  }
  const raw = wake.device.createBuffer({
    label: "wake:storage",
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const view = data.constructor as WakeArrayCtor<T>;
  new view(raw.getMappedRange()).set(data);
  raw.unmap();
  return new WakeBuffer<T>(wake, raw, view, true);
}

export function wrapBuffer<T extends WakeArray>(
  wake: Wake,
  raw: GPUBuffer,
  view: WakeArrayCtor<T>,
): WakeBuffer<T> {
  const requiredUsage = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  if ((raw.usage & requiredUsage) !== requiredUsage) {
    throw new Error("wake.wrap expects a buffer created with COPY_SRC and COPY_DST usage");
  }
  return new WakeBuffer<T>(wake, raw, view);
}
