import type { Wake } from "./wake";

export interface WakeArrays {
  f32: Float32Array;
  u32: Uint32Array;
}

export type WakeDtype = keyof WakeArrays;

export type WakeArray = WakeArrays[WakeDtype];

const BUFFER_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/** A typed array of f32 or u32 elements stored on the GPU. */
export class WakeBuffer<D extends WakeDtype = "f32"> {
  private readonly wake: Wake;
  private readonly owned: boolean;
  private destroyed = false;

  /** The underlying `GPUBuffer`. */
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

  /**
   * Copies a `Float32Array` to a new GPU buffer. Later changes to the array do not affect it.
   */
  static from(wake: Wake, data: Float32Array): WakeBuffer<"f32">;
  /**
   * Copies a `Uint32Array` to a new GPU buffer. Later changes to the array do not affect it.
   */
  static from(wake: Wake, data: Uint32Array): WakeBuffer<"u32">;
  /** Copies a typed array to a new GPU buffer. Later changes to the array do not affect it. */
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

  /** Creates a new zero-filled GPU buffer with the given length and dtype. */
  static empty<D extends WakeDtype>(wake: Wake, length: number, dtype: D): WakeBuffer<D> {
    const raw = wake.device.createBuffer({
      label: "wake:empty",
      size: length * 4,
      usage: BUFFER_USAGE,
    });
    return new WakeBuffer<D>(wake, raw, dtype, true);
  }

  /**
   * Wraps an existing `GPUBuffer` (without copying it) so WAKE can use it. Meant for apps that
   * already use WebGPU directly. The buffer needs STORAGE, COPY_SRC, and COPY_DST usage. You are
   * responsible for destroying the underlying `GPUBuffer` yourself, since destroying the
   * `WakeBuffer` leaves it untouched.
   */
  static wrap<D extends WakeDtype>(wake: Wake, raw: GPUBuffer, dtype: D): WakeBuffer<D> {
    return new WakeBuffer<D>(wake, raw, dtype, false);
  }

  get byteLength(): number {
    return this.length * 4;
  }

  /** Copies the buffer's contents back to JavaScript as a new typed array. */
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

  /** Overwrites the buffer's contents with a typed array of the same byte length. */
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

  /**
   * Sums all elements, returning a new one-element buffer containing the result. The result is 0
   * for an empty input buffer, and u32 sums wrap on overflow.
   */
  sum(): WakeBuffer<D> {
    return this.wake.sum(this);
  }

  /**
   * Finds the smallest element, returning a new one-element buffer containing the result. Throws
   * on an empty input buffer.
   */
  min(): WakeBuffer<D> {
    return this.wake.min(this);
  }

  /**
   * Finds the largest element, returning a new one-element buffer containing the result. Throws on
   * an empty input buffer.
   */
  max(): WakeBuffer<D> {
    return this.wake.max(this);
  }

  /**
   * Computes an exclusive prefix sum in place, returning the same buffer. Only u32 is supported.
   * Sums wrap on overflow.
   */
  scan(this: WakeBuffer<"u32">): WakeBuffer<"u32"> {
    return this.wake.scan(this);
  }

  /** Sorts in ascending order, in place, and returns the same buffer. The sort is stable. */
  sort(): WakeBuffer<D> {
    return this.wake.sort(this);
  }

  /**
   * Frees the GPU buffer. Safe to call more than once. Buffers created with `wrap()` are not
   * freed, and must be destroyed by you.
   */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.owned) this.raw.destroy();
  }
}
