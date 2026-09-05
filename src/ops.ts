import type { WakeBuffer, WakeSupportedArray } from "./buffer";

export type AssociativeOp = "sum" | "min" | "max";

export type WgslDtype = "f32" | "u32" | "i32";

export type AssociativeOpWgsl = {
  combine: string;
  identity: Record<WgslDtype, string>;
};

export const ASSOCIATIVE_OPS: Record<AssociativeOp, AssociativeOpWgsl> = {
  sum: {
    combine: "a + b",
    identity: { f32: "0f", u32: "0u", i32: "0i" },
  },
  min: {
    combine: "min(a, b)",
    identity: {
      f32: "0x1.fffffep+127f", // f32 max
      u32: "0xffffffffu",
      i32: "2147483647i",
    },
  },
  max: {
    combine: "max(a, b)",
    identity: {
      f32: "-0x1.fffffep+127f", // f32 min
      u32: "0u",
      i32: "-2147483647i - 1i",
    },
  },
};

export function wgslCombineFn(dtype: WgslDtype, op: AssociativeOpWgsl): string {
  return `
fn combine(a: ${dtype}, b: ${dtype}) -> ${dtype} {
  return ${op.combine};
}`;
}

export function wgslDtypeOf(buffer: WakeBuffer<WakeSupportedArray>): WgslDtype {
  if (buffer.view === Float32Array) return "f32";
  if (buffer.view === Uint32Array) return "u32";
  if (buffer.view === Int32Array) return "i32";
  throw new Error(
    `unsupported element type ${buffer.view.name}, expected Float32Array, Uint32Array, or Int32Array`,
  );
}
