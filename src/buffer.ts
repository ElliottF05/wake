import type { Wake } from "./wake";

export type WakeArray = Float32Array | Uint32Array | Int32Array;

type WakeView<T extends WakeArray> = {
  new (buffer: ArrayBufferLike): T;
  readonly BYTES_PER_ELEMENT: number;
};

class WakeBuffer<T extends WakeArray = Float32Array> {
  readonly #wake: Wake;
  #destroyed = false;

  readonly raw: GPUBuffer;
  readonly view: WakeView<T>;
  readonly length: number;
  readonly byteLength: number;

  constructor(wake: Wake, raw: GPUBuffer, view: WakeView<T>, length: number) {
    this.#wake = wake;
    this.raw = raw;
    this.view = view;
    this.length = length;
    this.byteLength = length * view.BYTES_PER_ELEMENT;
  }

  async read(): Promise<T> {
    if (this.#destroyed) {
      throw new Error("WakeBuffer.read called on a destroyed buffer");
    }
    const device = this.#wake.device;
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
    if (this.#destroyed) {
      throw new Error("WakeBuffer.write called on a destroyed buffer");
    }
    if (data.byteLength !== this.byteLength) {
      throw new Error(
        `WakeBuffer.write expected ${this.byteLength} bytes but received ${data.byteLength}`,
      );
    }
    this.#wake.device.queue.writeBuffer(this.raw, 0, data);
  }

  destroy(): void {
    if (this.#destroyed) return;
    this.#destroyed = true;
    this.raw.destroy();
  }
}

export type { WakeBuffer };

export function createBuffer<T extends WakeArray>(wake: Wake, data: T): WakeBuffer<T> {
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
  const view = data.constructor as WakeView<T>;
  new view(raw.getMappedRange()).set(data);
  raw.unmap();
  return new WakeBuffer<T>(wake, raw, view, data.length);
}
