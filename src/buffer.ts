import type { Wake } from "./wake";

export interface WakeArrays {
  f32: Float32Array;
  u32: Uint32Array;
}

export type WakeDtype = keyof WakeArrays;

export type WakeArray = WakeArrays[WakeDtype];

const BUFFER_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class WakeBuffer<D extends WakeDtype = "f32"> {
  private readonly wake: Wake;
  private readonly owned: boolean;
  private destroyed = false;

  readonly raw: GPUBuffer;
  readonly dtype: D;
  readonly length: number;

  private constructor(wake: Wake, raw: GPUBuffer, dtype: D, owned: boolean) {
    if ((raw.usage & BUFFER_USAGE) !== BUFFER_USAGE) {
      throw new Error("WakeBuffer requires a GPUBuffer with STORAGE, COPY_SRC, and COPY_DST usage");
    }
    this.wake = wake;
    this.owned = owned;
    this.raw = raw;
    this.dtype = dtype;
    this.length = raw.size / 4;
  }

  static from(wake: Wake, data: Float32Array): WakeBuffer<"f32">;
  static from(wake: Wake, data: Uint32Array): WakeBuffer<"u32">;
  static from(wake: Wake, data: WakeArray): WakeBuffer<WakeDtype>;
  static from(wake: Wake, data: WakeArray): WakeBuffer<WakeDtype> {
    let dtype: WakeDtype;
    if (data instanceof Float32Array) {
      dtype = "f32";
    } else if (data instanceof Uint32Array) {
      dtype = "u32";
    } else {
      throw new Error("WakeBuffer expects a Float32Array or Uint32Array");
    }
    const raw = wake.device.createBuffer({
      label: "wake:storage",
      size: data.byteLength,
      usage: BUFFER_USAGE,
      mappedAtCreation: true,
    });
    new Uint8Array(raw.getMappedRange()).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    );
    raw.unmap();
    return new WakeBuffer<WakeDtype>(wake, raw, dtype, true);
  }

  static empty<D extends WakeDtype>(wake: Wake, length: number, dtype: D): WakeBuffer<D> {
    const raw = wake.device.createBuffer({
      label: "wake:empty",
      size: length * 4,
      usage: BUFFER_USAGE,
    });
    return new WakeBuffer<D>(wake, raw, dtype, true);
  }

  static wrap<D extends WakeDtype>(wake: Wake, raw: GPUBuffer, dtype: D): WakeBuffer<D> {
    return new WakeBuffer<D>(wake, raw, dtype, false);
  }

  get byteLength(): number {
    return this.length * 4;
  }

  async read(): Promise<WakeArrays[D]> {
    if (this.destroyed) {
      throw new Error("attempted to read from a destroyed WakeBuffer");
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
      const data = this.dtype === "f32" ? new Float32Array(copy) : new Uint32Array(copy);
      return data as WakeArrays[D];
    } finally {
      staging.destroy();
    }
  }

  write(data: WakeArrays[D]): void {
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
