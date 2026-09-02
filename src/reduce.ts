import type { WakeArray, WakeBuffer } from "./buffer";
import type { Wake } from "./wake";

export type ReduceOperation = "sum"; // TODO: add other operations

// TODO: implement this
export async function reduce<T extends WakeArray>(
  wake: Wake,
  input: WakeBuffer<T>,
  op: ReduceOperation,
): Promise<WakeBuffer<T>> {
  throw new Error(`reduce: "${op}" is unimplemented`);
}
