import type { Wake } from "./wake";

export type WakeArray = Float32Array | Uint32Array | Int32Array;

export type WakeView<T extends WakeArray> = {
  new (buffer: ArrayBufferLike): T;
  readonly BYTES_PER_ELEMENT: number;
};

class WakeBuffer<T extends WakeArray = Float32Array> {
  readonly #wake: Wake;
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
    const device = this.#wake.device;
    const staging = device.createBuffer({
      size: this.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.raw, 0, staging, 0, this.byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return new this.view(copy);
  }

  write(data: T): void {
    if (data.byteLength !== this.byteLength) {
      throw new Error(
        `WakeBuffer.write expected ${this.byteLength} bytes but received ${data.byteLength}`,
      );
    }
    this.#wake.device.queue.writeBuffer(this.raw, 0, data);
  }

  destroy(): void {
    this.raw.destroy();
  }
}

export type { WakeBuffer };

export function createBuffer<T extends WakeArray>(wake: Wake, data: T): WakeBuffer<T> {
  const raw = wake.device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(raw.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  raw.unmap();
  return new WakeBuffer<T>(wake, raw, data.constructor as WakeView<T>, data.length);
}
