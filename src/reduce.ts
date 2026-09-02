import type { WakeArray, WakeBuffer } from "./buffer";
import type { Wake } from "./wake";

export type ReduceOperation = "sum"; // TODO: add other operations

// TODO: implement this
export async function reduce<T extends WakeArray>(
  wake: Wake,
  input: WakeBuffer<T>,
  op: ReduceOperation,
): Promise<WakeBuffer<T>> {
  // for now this just returns 0 always
  const out = new input.view(new ArrayBuffer(input.view.BYTES_PER_ELEMENT));
  return wake.buffer(out);
}
