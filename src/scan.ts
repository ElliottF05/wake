import type { WakeBuffer } from "./buffer";
import type { Wake } from "./wake";

export function scanScratchLength(n: number): number {
  throw new Error("not implemented");
}

export function scan(wake: Wake, input: WakeBuffer<"u32">): WakeBuffer<"u32"> {
  throw new Error("not implemented");
}

export function encodeScan(
  wake: Wake,
  pass: GPUComputePassEncoder,
  input: WakeBuffer<"u32">,
  scratch: WakeBuffer<"u32">,
): void {
  throw new Error("not implemented");
}
