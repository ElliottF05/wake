import type { WakeBuffer, WakeSupportedArray } from "./buffer";
import type { Wake } from "./wake";

export type ReduceOp = "sum"; // TODO: add other operations

type WgslScalarType = "f32" | "u32" | "i32";

// TODO: implement this
export async function reduce<T extends WakeSupportedArray>(
  wake: Wake,
  input: WakeBuffer<T>,
  op: ReduceOp,
): Promise<WakeBuffer<T>> {
  if (op !== "sum") {
    throw new Error(`unsupported operation "${op}"`);
  }
  // for now this just returns 0 always
  console.warn("reduction kernel not implemented, returning 0");
  const out = new input.view(new ArrayBuffer(input.view.BYTES_PER_ELEMENT));
  return wake.buffer(out);
}

function wakeBufferToWgslType(buffer: WakeBuffer<WakeSupportedArray>): WgslScalarType {
  if (buffer.view === Float32Array) return "f32";
  if (buffer.view === Uint32Array) return "u32";
  if (buffer.view === Int32Array) return "i32";
  throw new Error(`unsupported element type ${buffer.view.name}, expected Float32Array, Uint32Array, or Int32Array`);
}
